from __future__ import annotations

import typing as t
import functools
import weakref
from collections.abc import Iterator, Mapping
from pathlib import Path

from streamlit_process_manager.process import Process
from streamlit_process_manager import group, _core


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

    def __init__(self, process: Process, pgroup: "group.ProcessGroup" | None = None):
        """Create a ProcessProxy which weakly references a Process and optionally, it's containing Group."""
        self._proc_weak: weakref.ref[Process] = weakref.ref(process)
        self._pgroup_weak: weakref.ref["group.ProcessGroup"] | None = None if pgroup is None else weakref.ref(pgroup)
        self.supports_remove = self._pgroup_weak is not None
        """If True, this ProcessProxy supports calling `remove_from_pgroup`."""

    def _deref_proc(self, when="you are trying to perform an action on") -> Process:
        """Raise an exception if the process no longer exists."""
        if (deref := self._proc_weak()) is None:
            raise ValueError(
                f"The Process {when} no longer exists. It may have been removed in another session"
                "or the server may have been restarted."
            )
        return deref

    def _deref_pgroup(self, when="you are trying to perform an action on") -> "group.ProcessGroup" | None:
        """Raise an exception if the process group no longer exists."""
        if self._pgroup_weak is None:
            return None
        if (deref := self._pgroup_weak()) is None:
            raise ValueError(
                f"The ProcessGroup for the Process {when} no longer exists. It may have been removed in another session"
                "or the server may have been restarted."
            )
        return deref

    @staticmethod
    def _proxied_property(prop_name: str) -> t.Callable[[t.Callable[[t.Any, Process], T]], T]:
        """Do Voodo property magic so that we deref he process before we access a property on it."""

        def _decorator(func: t.Callable[[t.Any, Process], T]) -> T:

            @functools.wraps(func)
            def _proxprop(self: ProcessProxy) -> T:
                return func(self, self._deref_proc(when=f"you tried to get the {prop_name} of"))

            return t.cast("T", property(_proxprop))

        return _decorator

    def start(self) -> t.Self:
        """Start this process.

        If the process has already been started, raise a ChildProcessError.

        Returns: self so it can be chained after the constructor.
        """
        proc = self._deref_proc(when="you are trying to start")
        if (pgroup := self._deref_pgroup(when="you are trying to start")) is None or proc in pgroup._procs:
            proc.start()
            return self
        else:
            raise _core.UnsafeOperationError(
                "The Process is no longer part of its original ProcessGroup and cannot be started. It may have been "
                "moved or removed in another session."
            )

    def start_safe(self) -> t.Self:
        """Start this process, but if it's already started do nothing.

        Returns: self so it can be chained after the constructor.
        """
        # TODO: should this also discard errors from dereferencing?
        proc = self._deref_proc(when="you are trying to start")
        if (pgroup := self._deref_pgroup(when="you are trying to start")) is None or proc in pgroup._procs:
            proc.start_safe()
            return self
        else:
            raise _core.UnsafeOperationError(
                "The Process is no longer part of its original ProcessGroup and cannot be started. It may have been "
                "moved or removed in another session."
            )

    def terminate(self, wait_for_death: bool = False):
        """Call to force-kill this process.

        Parameters:
            wait_for_death (bool): Whether to block until the Process completes after terminating.
        """
        proc = self._deref_proc(when="you are trying to terminate")
        if (pgroup := self._deref_pgroup(when="you are trying to terminate")) is None or proc in pgroup._procs:
            return proc.terminate(wait_for_death=wait_for_death)
        else:
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
        else:
            raise _core.UnsafeOperationError(
                "The Process is no longer part of its original ProcessGroup and cannot be interrupted. It may have been"
                " moved or removed in another session."
            )

    def remove_from_pgroup(self) -> None:
        """Remove this Process from it's ProcessGroup (if it exists).

        If the process is already removed from it's group, or this is a proxy which does not reference a group,
        then this function does nothing.
        """
        if (pgroup := self._deref_pgroup(when="you are trying to remove")) is None:
            return None  # do nothing if this process is not part of a group
        try:
            return pgroup.remove(self._deref_proc(when="you are trying to remove"))
        except ValueError:
            return None  # do nothing if the process is not in the group t.Anymore

    def peek_output(self, nlines: int | None = None) -> list[str]:
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

    def monitor(self, nlines: int | None = 10) -> Iterator[list[str]]:
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
            else:
                yield output

    def to_dict(self) -> Process.SavedProcessDict:
        """Convert this Process into a dictionary representation."""
        return self._deref_proc(when="trying to convert this process to a dict").to_dict()

    def close_output(self):
        """Close the output_file on this Process."""
        if (_proc := self._proc_weak()) is not None:
            _proc.close_output()

    # pylint doesn't like our magical property decorator so pylint: disable=property-with-parameters

    @_proxied_property("start time")
    def _start_time(self, proc: Process) -> float | None:  # noqa: D401
        """The start time of this Process, or None if it is not started."""
        return proc._start_time

    @_proxied_property("command")
    def cmd(self, proc: Process) -> list[str]:  # noqa: D401
        """A copy of the command and args used for this Process."""
        return proc.cmd

    @_proxied_property("environment")
    def env(self, proc: Process) -> Mapping[str, str] | None:  # noqa: D401
        """A copy of the environment variables used for this Process."""
        return proc.env

    @_proxied_property("pid")
    def pid(self, proc: Process) -> int | None:  # noqa: D401
        """The Process ID of this Process (if it has started) otherwise None."""
        return proc.pid

    @_proxied_property("returncode")
    def returncode(self, proc: Process) -> int | None:  # noqa: D401
        """The returncode from this process.

        Returns:
            int | None: The returncode, or None if the process has not yet finished.
        """
        return proc.returncode

    @_proxied_property("state")
    def state(self, proc: Process) -> str | None:  # noqa: D401
        """The state of this Process (if it has started) otherwise None."""
        return proc.state

    @_proxied_property("started")
    def started(self, proc: Process) -> bool:  # noqa: D401
        """Whether or not this Process has been started."""
        return proc.started

    @_proxied_property("finished")
    def finished(self, proc: Process) -> bool:  # noqa: D401
        """Whether or not this Process has finished."""
        return proc.finished

    @_proxied_property("running")
    def running(self, proc: Process) -> bool:  # noqa: D401
        """Whether or not this Process is currently running."""
        return proc.running

    @_proxied_property("time since start")
    def time_since_start(self, proc: Process) -> float | None:  # noqa: D401
        """The number of seconds since this Process started running, or None if it hasn't been started."""
        return proc.time_since_start

    @_proxied_property("ability to be started")
    def can_be_started(self, proc: Process) -> bool:  # noqa: D401
        """Whether or not this Process can be started."""
        return proc.can_be_started

    @_proxied_property("ability to be interrupted")
    def can_be_interrupted(self, proc: Process) -> bool:  # noqa: D401
        """Whether or not this Process can be interrupted."""
        return proc.can_be_interrupted

    @_proxied_property("output_file")
    def output_file(self, proc: Process) -> Path | None:  # noqa: D401
        """The output file chosen for this process."""
        return proc.output_file

    @_proxied_property("output_encoding")
    def output_encoding(self, proc: Process) -> str:  # noqa: D401
        """The output encoding chosen for this process."""
        return proc.output_encoding

    @_proxied_property("label")
    def label(self, proc: Process) -> str:  # noqa: D401
        """The label assigned to this Process."""
        return proc.label

    def is_broken(self):
        """Return true if the Process this proxy points to no longer exists."""
        return self._proc_weak() is None

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