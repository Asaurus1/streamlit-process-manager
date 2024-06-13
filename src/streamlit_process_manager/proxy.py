"""Module for ProcessProxy.

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

import typing as t
import weakref
from collections.abc import Iterator

from streamlit_process_manager.process import Process
from streamlit_process_manager import _core

if t.TYPE_CHECKING:
    from streamlit_process_manager.types import Self
    from streamlit_process_manager import group
    from pathlib import Path


T = t.TypeVar("T")


class ProcessProxy:
    """A proxy to a Process.

    This is typically returned by a ProcessGroup in order to provide protection against concurrency issues with
    multiple user sessions. It contains weak references to an underlying Process object (as well as an optional
    ProcessGroup object for performing "remove" operations) and will only allow certain actions like start or
    interrupt on processes that are still within the referenced group / still in exsistence in memory.
    """

    # we do this all over this proxy so pylint: disable=protected-access
    # and like Process it proxies, this needs pylint: disable=too-many-public-methods

    def __init__(self, process: Process, pgroup: "group.ProcessGroup | None" = None):
        """Create a ProcessProxy which weakly references a Process and optionally, it's containing Group."""
        self._proc_weak: weakref.ref[Process] = weakref.ref(process)
        self._pgroup_weak: "weakref.ref[group.ProcessGroup] | None" = None if pgroup is None else weakref.ref(pgroup)
        self.supports_remove = self._pgroup_weak is not None
        """If True, this ProcessProxy supports calling `remove_from_pgroup`."""

    def _deref_proc(self, when="you are trying to perform an action on") -> Process:
        """Raise an exception if the process no longer exists."""
        if (deref := self._proc_weak()) is None:
            raise ValueError(
                f"The Process {when} no longer exists. It may have been removed in another session "
                "or the server may have been restarted."
            )
        return deref

    def _deref_pgroup(self, when="you are trying to perform an action on") -> "group.ProcessGroup | None":
        """Raise an exception if the process group no longer exists."""
        if self._pgroup_weak is None:
            return None
        if (deref := self._pgroup_weak()) is None:
            raise ValueError(
                f"The ProcessGroup for the Process {when} no longer exists. It may have been removed in another session"
                "or the server may have been restarted."
            )
        return deref

    def start(self) -> "Self":
        """Start this process.

        If the process has already been started, raise a ChildProcessError.

        Returns: self so it can be chained after the constructor.
        """
        proc = self._deref_proc(when="you are trying to start")
        if (pgroup := self._deref_pgroup(when="you are trying to start")) is None or proc in pgroup._procs:
            proc.start()
            return self
        raise _core.UnsafeOperationError(
            "The Process is no longer part of its original ProcessGroup and cannot be started. It may have been "
            "moved or removed in another session."
        )

    def start_safe(self) -> "Self":
        """Start this process, but if it's already started do nothing.

        Returns: self so it can be chained after the constructor.
        """
        # TODO: should this also discard errors from dereferencing?
        proc = self._deref_proc(when="you are trying to start")
        if proc.can_be_started:
            return self.start()
        # else
        return self

    def terminate(self, wait_for_death: bool = False):
        """Call to force-kill this process.

        Parameters:
            wait_for_death (bool): Whether to block until the Process completes after terminating.
        """
        proc = self._deref_proc(when="you are trying to terminate")
        if (pgroup := self._deref_pgroup(when="you are trying to terminate")) is None or proc in pgroup._procs:
            return proc.terminate(wait_for_death=wait_for_death)
        raise _core.UnsafeOperationError(
            "The Process is no longer part of its original ProcessGroup and cannot be terminated. It may have been "
            "moved or removed in another session."
        )

    def interrupt(self, wait_for_death: bool = True, force: bool = False):
        """Call to interrupt this process.

        Parameters:
            wait_for_death (bool): Whether to block until the Process completes after interrupting.
            force (bool): Whether to force a terminate if the Process cannot be safely interrupted.
                - Most processes can be interrupted as if you had pressed Ctrl-C; however on Windows processes which
                  are created by the special classmethods like `from_existing` or `from_pid` cannot be safely
                  interrupted without potentially killing the calling process as well. For these processes, an
                  UnsafeOperationError will be raised unless "force" is set to True.

        """
        proc = self._deref_proc(when="you are trying to interrupt")
        if (pgroup := self._deref_pgroup(when="you are trying to interrupt")) is None or proc in pgroup._procs:
            return proc.interrupt(wait_for_death=wait_for_death, force=force)
        raise _core.UnsafeOperationError(
            "The Process is no longer part of its original ProcessGroup and cannot be interrupted. It may have been"
            " moved or removed in another session."
        )

    def remove_from_pgroup(self) -> None:
        """Remove this Process from it's ProcessGroup (if it exists).

        If the process is already removed from it's group, or this is a proxy which does not reference a group,
        then this function does nothing.
        """
        try:
            if (pgroup := self._deref_pgroup(when="you are trying to remove")) is None:
                return None  # do nothing if this process is not part of a group
            return pgroup.remove(self._deref_proc(when="you are trying to remove"))
        except ValueError:
            return None  # do nothing if the process is not in the group anymore or the group is gone

    def peek_output(self, nlines: "int | None" = None) -> t.List[str]:
        """Get up to the last nlines of output from the process, as a list of strings.

        Parameters:
            nlines (optional, int): The maximum number of lines to return. If None, all lines are returned.

        Returns:
            list[str]: The last nlines of output (or fewer) from the monitored output_file.
                Strings are newline-terminated, and so may be joined with `"".join(proc.peek_output())` to recreate
                the file contents. If the process is not running or the output file does not exist, an empty list is
                returned.
        """
        return self._deref_proc(when="you are trying to peek_output on").peek_output(nlines=nlines)

    def monitor(self, nlines: "int | None" = 10) -> Iterator[t.List[str]]:
        """Get an Iterator that yields the last nlines from the output_file, and completes when the Process is finished.

        Parameters:
            nlines (int): number of lines of output to show.

        Returns:
            list[str]: A list of strings representing the last nlines of output from this Process.
        """
        while self.started:
            output = self.peek_output(nlines=nlines)
            if self.finished:  # pylint: disable=using-constant-test
                yield output
                return
            # else
            yield output

    def to_dict(self) -> Process.SavedProcessDict:
        """Convert this Process into a dictionary representation."""
        return self._deref_proc(when="trying to convert this process to a dict").to_dict()

    def close_output(self):
        """Close the output_file on this Process."""
        if (_proc := self._proc_weak()) is not None:
            _proc.close_output()

    @property
    def is_broken(self):
        """Return true if the Process this proxy points to no longer exists."""
        return self._proc_weak() is None

    # Typing for standard attributes that can be accessed via __getattr__
    _start_time: "float | None"
    cmd: t.List[str]
    env: t.Dict[str, str]
    pid: "int | None"
    returncode: "int | None"
    state: "str | None"
    started: bool
    finished: bool
    running: bool
    time_since_start: "float | None"
    can_be_started: bool
    can_be_interrupted: bool
    output_file: "Path | None"
    output_encoding: str
    label: str

    def __getattr__(self, name: str):
        """Access an attribute on the proxied Process that isn't otherwise defined."""
        return self._deref_proc(when=f"trying to access the '{name}' attribute").__getattribute__(name)

    def __setattr__(self, name: str, value):
        """Set attributes on the proxy object generally fails."""
        if name not in ("_proc_weak", "_pgroup_weak", "supports_remove"):
            raise AttributeError("Cannot set attributes of a process via a Proxy object")
        if name == "supports_remove" and hasattr(self, "supports_remove"):  # supports_remove can only be set once
            raise AttributeError("supports_remove is read-only")
        object.__setattr__(self, name, value)

    def __eq__(self, other: object) -> bool:
        """Check if this Process is equal to another."""
        my_proc = self._proc_weak()
        try:
            # pylint: disable=protected-access
            other_proc = t.cast(ProcessProxy, other)._proc_weak()
            if other_proc is None:
                return False  # Broken ProcessProxies are always not equal to each other.
            return other_proc == my_proc
        except AttributeError:
            return my_proc == t.cast(Process, other)  # other might be a true Process
