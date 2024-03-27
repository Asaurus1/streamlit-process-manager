from __future__ import annotations

import io
import typing as t
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path

import yaml

from streamlit_process_manager.process import Process
from streamlit_process_manager.group import ProcessGroup
from streamlit_process_manager.proxy import ProcessProxy

stu = None  # temp

class ProcessManager:
    """Singleton manager class for processes."""

    def __init__(self, cachefile: str | Path):
        """Create a ProcessManager which references the specified cachefile."""
        self._groups: dict[str, ProcessGroup] = defaultdict(ProcessGroup)
        "Process group storage, by group name."
        self._single_groups: set[str] = set()
        "Set of groups which were created as 'single'."
        self._cachefilehandle = self._setup_cachefile(cachefile)

    def _setup_cachefile(self, cachepath) -> io.TextIOWrapper:
        """Create and open this manager's cachefile. Returns the Stream object for the file."""
        cachepath = Path(cachepath)
        # Open the cache file for both reading and writing, but create it if it does not exist.
        filemode = "r+" if cachepath.exists() else "w+"
        # we want to hold open the file as long as this singleton exists so pylint: disable=consider-using-with
        return cachepath.open(filemode, encoding="utf-8")

    def __del__(self):
        """Request that currently open processes terminate but do not wait for them to die."""
        # TODO: this may not actually be what we want to do long-term but is generally
        # correct since if this class is dying it either means we got cache cleared or
        # streamlit is shutting down.
        try:
            for pg in self._groups.values():
                pg.terminate_all(wait_for_death=False)
        finally:
            try:
                if not self._cachefilehandle.closed:
                    self._write_to_disk()
            finally:
                try:
                    self._cachefilehandle.close()
                except AttributeError:  # pragma: no cover
                    pass  # ignore if _cachefilehandle doesn't exist or doesn't have a .close method()

    @property
    def groups(self) -> list[str]:
        """Return a list of ProcessGroups in this ProcessManager."""
        return list(self._groups.keys())

    def group(self, key: str) -> ProcessGroup:
        """Return the ProcessGroup referenced by the specified name."""
        return self._groups[key]

    @t.overload
    def add(self, process: Process, group: str, start: bool = False) -> ProcessProxy:  # noqa: D102
        ...

    @t.overload
    def add(self, process: Iterable[Process], group: str, start: bool = False) -> list[ProcessProxy]:  # noqa: D102
        ...

    def add(self, process: Process | Iterable[Process], group: str, start: bool = False):
        """Add one ore more Processes to the named group.

        Parameters:
            process (Process or Iterable[Process]): One or more processes to add to the named group.
            group (str): The name of the group to add the processes to. Group cannot previously have been used
              with `.single()`.
            start (bool): Specify True to start the processes as you add them if they are not already running.

        Returns:
            Process | list[Process]:
        """
        if group in self._single_groups:
            raise ValueError(
                f"ProcessGroup {group} was previously populated using the .single() method. Please use "
                "that to add a process to this group instead."
            )
        return self._add(process, group, start=start)

    def _add(self, process: Process | Iterable[Process], group: str, start: bool = False):
        """Add one or more Processes to the named group without checking for 'single'ness."""
        if isinstance(process, Process) or not hasattr(process, "__iter__"):
            _procs: Iterable[Process] = [process]  # type: ignore[list-item]
        else:
            _procs = process
        _proxies: list[ProcessProxy] = []
        for proc in _procs:
            self._groups[group].add(process=proc)
            if start and proc.can_be_started:
                proc.start()
            _proxies.append(ProcessProxy(proc, pgroup=self._groups[group]))
        self._write_to_disk()
        if len(_proxies) == 1:
            return _proxies[0]
        else:
            return _proxies

    def single(self, process: Process, group: str | None = None, start: bool = False) -> ProcessProxy:
        """Add a single process to a named group, or return the existing Process from that group.

        Parameters:
            process (Process): The process to add.
            group (str): The name of a group to add the process to.
                - If the named group already contains a process, then the one passed via the "process" argument is
                discared.
                - If the named group already contains more than one process, a ValueError is raised.
                - If you are running in a Streamlit environment, then the group name is optional and the default value
                is the ID of the current Streamlit session.

        Returns:
            Process: Either the process passed in via "process" (if it was added to the group) or the already extant
                process inside of the named group.
        """
        if group is None:
            from streamlit.runtime.scriptrunner import get_script_run_ctx  # lazy import this

            if (script_ctx := get_script_run_ctx()) is None:
                raise RuntimeError("Cannot create single process without streamlit session context and no 'group' arg.")
            _group: str = "single_group_for_session_" + script_ctx.session_id
        else:
            _group = group
        pg: ProcessGroup = self.group(_group)
        group_len = len(pg)

        if group_len > 1:
            raise ValueError(
                f"Cannot create/get single process for group {_group} as the group already exists "
                "and has more than one process in it."
            )
        if group_len == 0:
            _proc: ProcessProxy = t.cast(ProcessProxy, self._add(process=process, group=_group, start=False))
        else:
            _proc = pg[0]

        if start and not _proc.running:
            _proc.start()

        return _proc

    def to_dict(self, groups: Sequence[str] | None = None) -> dict[str, list[Process.SavedProcessDict]]:
        """Convert the current state of this ProcessMonitor and all contained Processes to a dict.

        Returns:
            dict: representation of all groups and their Processes.
        """
        if groups is None:
            groups_to_write = self._groups
        else:
            groups_to_write = {group: self.group(group) for group in groups if group in self._groups}

        return {group_name: pg.to_dicts() for group_name, pg in groups_to_write.items()}

    def _write_to_disk(self) -> None:
        """Write the ProcessManager's current state to the cachefile."""
        self._cachefilehandle.truncate(0)
        self._cachefilehandle.seek(0)
        yaml.safe_dump(self.to_dict(), self._cachefilehandle)

    def _read_from_disk(self) -> None:
        """Read the ProcessManager's state from disk (not currently used)."""
        self._cachefilehandle.seek(0)
        data: dict[str, list[Process.SavedProcessDict]] = yaml.safe_load(self._cachefilehandle)
        if not isinstance(data, dict):
            raise ValueError(f"Bad cache data in {self._cachefilehandle}")
        for group_name, pg_data in data.items():
            self._groups[group_name] = ProcessGroup([Process.from_dict(process_data) for process_data in pg_data])