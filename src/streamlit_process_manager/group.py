"""Module for ProcessGroup.

Copyright 2024 Alexander Martin

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import annotations

import threading
import typing as t
from collections.abc import Iterable, Iterator, Sequence  # pylint: disable=unused-import

from streamlit_process_manager.process import Process
from streamlit_process_manager import proxy, _core

if t.TYPE_CHECKING:
    from streamlit_process_manager.types import ProcessOrProxy


class ProcessGroup(Sequence):
    """Container for multiple Process objects."""

    def __init__(self, procs: "Iterable[Process] | None" = None):
        """Create a new group of Processes from the specified Iterable.

        Parameters:
            procs (Iterable[Process]): An iterable of Processes. All processes must be unique, or a ValueError is raised
        """
        if procs is not None:
            procs_list = list(procs)
            if len(set(procs_list)) != len(procs_list):
                raise ValueError("Not all input Processes are unique")
            self._procs: t.List[Process] = procs_list
        else:
            self._procs = []

        self._lock = threading.Lock()

    def add(self, process: "ProcessOrProxy"):
        """Add the provided Process to this group.

        Parameters:
            process (Process or ProcessProxy): the Process to add. If it already exists in this group,
            a ValueError is raised.

        Returns:
            ProcessProxy: a new proxy for the Process just added.
        """
        if process in self._procs:
            raise ValueError("Process already exists in ProcessGroup, cannot add twice.")
        if isinstance(process, proxy.ProcessProxy):
            # pylint: disable=protected-access
            _process: Process = process._deref_proc(when="you are trying to add to a ProcessGroup")
        else:
            _process = process
        self._procs.append(_process)
        return proxy.ProcessProxy(_process, pgroup=self)

    def start_all(self):
        """Start all processes in this group that can be started."""
        with self._lock:
            for proc in self._procs:
                proc.start_safe()

    def pop_finished(self) -> t.List[Process]:
        """Remove all finished processes from this group and return them as a list."""
        poplist = []
        newprocs = []
        with self._lock:
            for process in self._procs:
                if process.finished:
                    poplist.append(process)
                else:
                    newprocs.append(process)
            self._procs = newprocs
        return poplist

    def terminate_all(self, wait_for_death: bool = False):
        """Terminate all processes in this group.

        If exceptions are encountered during termination, the last exception encountered is re-raised.

        Parameters:
            wait_for_death (bool): whether to block until the process terminates.
        """
        last_err = None
        with self._lock:
            for process in self._procs:
                try:
                    process.terminate(wait_for_death=wait_for_death)
                except Exception as exc:  # pylint: disable=broad-exception-caught
                    # If there's any issues during termination, keep going because we want to make sure all the
                    # processes at least try.
                    last_err = exc
        if last_err:
            # If we did encounter at least one error, though, raise it here.
            raise last_err

    def interrupt_all(self, wait_for_death: bool = True, force: bool = False):
        """Interrupt all processes in this group.

        If exceptions are encountered during interruption, the last exception encountered is re-raised.

        Parameters:
            wait_for_death (bool): whether to block until the process stops.
            force (bool): whether to terminate processes which are unabled to be interrupted.
        """
        last_err = None
        with self._lock:
            for process in self._procs:
                try:
                    process.interrupt(wait_for_death=wait_for_death, force=force)
                except Exception as exc:  # pylint: disable=broad-exception-caught
                    # If there's any issues during interruption, keep going because we want to make sure all the
                    # processes at least try.
                    last_err = exc
        if last_err:
            # If we did encounter at least one error, though, raise it here.
            raise last_err

    def unsafe_clear(self):
        """Clear all Processes from this group.

        This does not return any of the Processes so if you want to retain references to them, use
        `saved_procs = group.procs` first. This is also not safe to call in general, as it does not check
        whether any of the processes in the group are running before clearing them. To remove all processes safely,
        iterate over the processes in the group and call `.remove()` on them after determining whether you want to
        remove them based on their state.
        """
        with self._lock:
            self._procs.clear()

    def remove(self, proc: "ProcessOrProxy"):
        """Remove the specified Process from this group.

        If the Process is running, it cannot be removed and will raise an UnsafeOperationError.
        """
        with self._lock:
            if proc in self._procs and proc.running:
                raise _core.UnsafeOperationError(
                    "The process you are attempting to remove from this group is running "
                    "and cannot be removed safely. Please stop the process before removing."
                )
            self._procs.remove(t.cast(Process, proc))

    def to_dicts(self) -> t.List[Process.SavedProcessDict]:
        """Get a snapshot of all Processes in this group as SavedProcessDicts."""
        with self._lock:
            return [proc.to_dict() for proc in self._procs]

    @property
    def is_empty(self) -> bool:
        """True if there are no Processes in this group."""
        return len(self._procs) == 0

    @property
    def labels(self) -> t.List[str]:
        """A list of Process labels from this ProcessGroup."""
        with self._lock:
            return [proc.label for proc in self._procs]

    @property
    def returncodes(self) -> t.List["int | None"]:
        """A list of Process returncodes from this ProcessGroup."""
        with self._lock:
            return [proc.returncode for proc in self._procs]

    def by_label(self, label: str, match_whole: bool = True) -> t.List["proxy.ProcessProxy"]:
        """Return a list of proxies for the Processes from this group with a label that matches the specified string.

        Parameters:
            label (str): the string to match against.
            match_whole (bool): whether the Process label must exactly match the provided string, or only a part.

        Returns:
            t.List[Process]: a list of matching processes.
        """
        with self._lock:
            return [proxy.ProcessProxy(proc, pgroup=self) for proc in self._procs if _match_condition(proc.tags)]

    @property
    def procs(self) -> t.List[proxy.ProcessProxy]:
        """Return a list of proxies for the Processes in this group."""
        with self._lock:
            return [proxy.ProcessProxy(proc, pgroup=self) for proc in self._procs]

    @t.overload
    def __getitem__(self, index: int) -> proxy.ProcessProxy:  # noqa: D105
        ...

    @t.overload
    def __getitem__(self, _slice: slice) -> t.List[proxy.ProcessProxy]:  # noqa: D105
        ...

    def __getitem__(self, index):
        """Return a proxy for the Process at the specified index from this ProcessGroup."""
        return proxy.ProcessProxy(self._procs[index], pgroup=self)

    def __iter__(self) -> Iterator[proxy.ProcessProxy]:
        """Iterate over proxies Processes in this ProcessGroup.

        This iterator represents the processes in this group at the moment it is called. Other threads may subsequently
        modify the ProcessGroup object but this will not be reflected in the iterator.
        """
        return iter(self.procs)

    def __len__(self) -> int:
        """Return the number of Processes in this ProcessGroup."""
        return len(self._procs)

    def __contains__(self, proc: object) -> bool:
        """Return true if this object contains the specified Process or ProcessProxy."""
        with self._lock:
            return proc in self._procs
