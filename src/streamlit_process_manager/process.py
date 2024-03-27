from __future__ import annotations

import io
import signal
import subprocess
import time
import typing as t
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path

import cachetools.func
import psutil

from streamlit_process_manager import _core


T = t.TypeVar("T")


class Process:
    """Represents a runnable subprocess (that may or may not have been started)."""

    # Refactoring this would increase complexity without adding value pylint: disable=too-many-instance-attributes
    # Because we have a lot of simple properties, pylint: disable=too-many-public-methods

    POLL_CACHE_TTL = 0.08
    "Time, in seconds, to retain a cache of the .poll() value for a process."

    def __init__(
        self,
        cmd: Iterable[str],
        output_file: Path | str | None,
        *,
        output_encoding: str = "utf-8",
        capture: t.Literal["none", "stderr", "stdout", "all"] = "all",
        env: Mapping[str, str] | None = None,
        label: str | None = None,
        cache_output_capture: bool = True,
    ):
        """Create a Process object (does not start the process).

        Parameters:
            cmd (Iterable[str]): A sequence of strings representing the command to be run. Shell process commands
                are NOT supported due to the complexities of terminating shell processes.
            output_file (PathLike): A filepath representing an output file to monitor. The Popen process must already
                be configured to output to this file.
            output_encoding (optional, t.Any valid string encoding): Which encoding to use when reading the output file,
                defaults to UTF-8.
            capture (one of "none", "stderr", "stdout", "all"): Which output streams, if t.Any, to redirect to
                output_file in the created process. Defaults to "all".
            env (Mapping[str, str]): A mapping/dict representing the environment to send to the current process.
                Defaults to the current process's environment if not specified.
            label (optional, str): A label to assign to this process. Defaults to the command line arguments.
            cache_output_capture (optional, bool): Whether to cache lines read from the output file.

        Returns:
            Process: A new Process object. The specified cmd will not be run as a subprocess until `.start()` is called.
        """
        # Private Members. These should be initialized first because they are referenced by public members and by the
        # __del__ function, and if there's a problem later we still want to be able to deconstruct the object.
        self._proc: psutil.Process | subprocess.Popen | None = None
        "The internal process handle."
        self._output_filehandle: io.TextIOWrapper | None = None
        "An optional read-only filehandle to the process's output_file."
        self._output_buffer: list[str] = []
        "Stores lines read from the output file."
        self._env: dict[str, str] | None
        "Internal env store."
        self._cmd: list[str]
        "Internal cmd store."
        self._start_time: float | None = None
        "Value of time.time() when the process was started."
        self._can_be_interrupted: bool = True
        """Whether or not the process can be interrupted rather than terminated.

        On Windows, this is only true for processes created by .start() since we cannot guarantee that external
        processes were created with a separate ProcessGroup and won't kill the current process when issuing a
        BREAK signal."""

        # Public properties
        self.cmd = cmd  # type: ignore[assignment]  # https://github.com/python/mypy/issues/3004
        self.env = env

        # Public members
        self.output_file: Path | None = Path(output_file) if output_file is not None else None
        "A path to a file where the process output should be stored. If no output is expected/desired, may be None."
        self.output_encoding: str = output_encoding
        "Default encoding for the output file. Disregarded if output_file is None."
        self._capture_settings: tuple[bool, bool] = (capture in ("all", "stdout"), capture in ("all", "stderr"))
        "Whether to capture STDOUT/STDERR from the process."
        if capture not in ("all", "stderr", "stdout", "none"):
            raise ValueError(f"Invalid capture setting '{capture}'. Expected 'all', 'stderr', 'stdout', or 'none'.")
        if any(self._capture_settings) and self.output_file is None:
            raise ValueError("Cannot request output capture without setting an output_file for the process")
        self.cache_output_capture: bool = cache_output_capture
        """Whether or not to cache output lines read from the file.

        Disable this if you expect the process to rewrite or delete lines in the output file, rather than
        simply write in "append" mode. Disabling will decrease performance for
        large output files. Ignored if output_file is None."""
        self.label: str = _default_label_if_unset(label, self)
        "A label for the process."

    def start(self) -> t.Self:
        """Start the process.

        If the process has already been started, raise a ChildProcessError.

        Returns: self so it can be chained after the constructor.
        """
        if self.started:
            raise ChildProcessError("Cannot start a process that has alread started.")

        if self.output_file is None:
            self._start_process()
        else:
            with self.output_file.open("w") as f:  # pylint: disable=unspecified-encoding
                self._start_process(f)
        return self

    def start_safe(self) -> t.Self:
        """Start the process if it can be started, otherwise do nothing."""
        if self.can_be_started:
            return self.start()
        else:
            return self

    def _start_process(self, f=None):
        """Start the internal _proc (Popen)."""
        stdout_dest, stderr_dest = self._get_output_destinations(f)
        cflags = 0  # subprocess.CREATE_NEW_PROCESS_GROUP if psutil.WINDOWS else 0, see .interrupt()
        self._proc = psutil.Popen(self.cmd, stdout=stdout_dest, stderr=stderr_dest, env=self.env, creationflags=cflags)
        self._can_be_interrupted = not psutil.WINDOWS  # True can't be used here; see .interrupt()
        self._start_time = self._proc.create_time()
        self._poll.cache_clear()

    def terminate(self, wait_for_death=False):
        """Call to force-kill the process.

        Parameters:
            wait_for_death (bool): Whether to block until the Process completes after terminating.
        """
        if self._proc is None:
            return
        self._raise_if_proc_not_child(action="terminate")
        try:
            self._proc.terminate()
            self._poll.cache_clear()
            if wait_for_death:
                self._proc.wait()
        except psutil.NoSuchProcess:
            pass

    def interrupt(self, wait_for_death=True, force=False):
        """Call to interrupt the process.

        Parameters:
            wait_for_death (bool): Whether to block until the Process completes after interrupting.
            force (bool): Whether to force a terminate if the Process cannot be safely interrupted.
                - Most processes can be interrupted as if you had pressed Ctrl-C; however on Windows processes which
                  are created by the special classmethods like `from_existing` or `from_pid` cannot be safely
                  interrupted without potentially killing the calling process as well. For these processes, an
                  UnsafeOperationError will be raised unless "force" is set to True.

        """
        if self._proc is None:
            return None

        if not self.can_be_interrupted:
            if force:
                return self.terminate(wait_for_death=wait_for_death)
            else:
                raise _core.UnsafeOperationError(
                    "This process cannot be safely interrupted on your platform. Use .terminate() instead or set "
                    "'force=True'"
                )

        self._raise_if_proc_not_child(action="interrupt")
        try:
            # if psutil.WINDOWS:
            #     self._proc.send_signal(signal.CTRL_BREAK_EVENT)  # on linux, pylint: disable=no-member
            # else:  # LINUX
            #     self._proc.send_signal(signal.SIGINT)
            # In order to make this work on windows, we have to have the process be in a separate group and then it
            # doesn't get closed by streamlit normally terminating. So we don't do it.
            self._proc.send_signal(signal.SIGINT)
            self._poll.cache_clear()
            if wait_for_death:
                self._proc.wait()
        except psutil.NoSuchProcess:
            pass
        return None

    def _raise_if_proc_not_child(self, action: str):
        """Raise an UnsafeOperationError if this Process is not a child of the current process."""
        if (
            not _core.UNSAFE_ALLOW_NONCHILDREN_TERMINATION
            and (pid := self.pid) is not None
            and self.running
            and not _is_pid_child_of_current(pid)
        ):
            raise _core.UnsafeOperationError(f"Cannot {action} process {pid} which is not a child of the current process.")

    class SavedProcessDict(t.TypedDict):
        """Represents a Process that has been saved to disk."""

        pid: int | None
        cmd: list[str]
        label: str
        state: str | None
        env: Mapping[str, str] | None
        output_file: str | None
        output_encoding: str
        start_time: float | None
        returncode: int | None

    def to_dict(self) -> SavedProcessDict:
        """Convert this Process object to a SavedProcessDict representation."""
        return Process.SavedProcessDict(
            pid=self.pid,
            cmd=self.cmd,
            label=self.label,
            state=self.state,
            env=self.env,
            output_file=str(self.output_file),
            output_encoding=self.output_encoding,
            start_time=self._start_time,
            returncode=self.returncode,
        )

    @classmethod
    def from_dict(cls, data: SavedProcessDict) -> t.Self | FinalizedProcess:
        """Create a new process from a SavedProcessDict.

        Parameters:
            data (SavedProcessDict): A mapping containg the info needed to reconstruct the process.

        Returns:
            Process | t.Finalized Process: If the process specified in the SavedProcessDict is still running,
            then a Process will be returned and can be controlled as usual. If the process specified in the
            SavedProcessDict is no longer running or cannot be found, a FinalizedProcess will be returned
            which represents the last-known state of the Process.
        """
        maybe_new_proc: t.Self | FinalizedProcess | None
        new_proc: t.Self | FinalizedProcess

        if data["pid"] is not None:
            finalize_because: str = ""
            try:
                maybe_new_proc = cls.from_pid(
                    pid=data["pid"],
                    output_file=data["output_file"],
                    output_encoding=data["output_encoding"],
                    label=data["label"],
                )
            except psutil.NoSuchProcess:
                maybe_new_proc = None
                finalize_because = f"Process with pid {data['pid']} no longer exists."

            if maybe_new_proc is not None:
                allowed_start_time = data["start_time"]
                allowed_env = data["env"]
                # pylint: disable=protected-access
                if allowed_env is not None and maybe_new_proc.env != _marshall_env_dict(allowed_env):
                    maybe_new_proc = None
                    finalize_because = f"Process with pid {data['pid']} has an unexpected environment."
                elif allowed_start_time is not None and allowed_start_time != maybe_new_proc._start_time:
                    maybe_new_proc = None
                    finalize_because = f"Process with pid {data['pid']} has an unexpected create time."

            if maybe_new_proc is None:
                new_proc = FinalizedProcess(
                    cmd=data["cmd"],
                    output_file=data["output_file"],
                    output_encoding=data["output_encoding"],
                    label=data["label"],
                    returncode=data.get("returncode"),
                    finalized_because=finalize_because,
                )
            else:
                new_proc = maybe_new_proc
        else:
            new_proc = cls(
                cmd=data["cmd"],
                output_file=data["output_file"],
                output_encoding=data["output_encoding"],
                label=data["label"],
                env=data["env"],
            )
        new_proc._cmd = data["cmd"]  # pylint: disable=protected-access

        return new_proc

    @classmethod
    def from_existing(
        cls,
        proc: psutil.Popen | subprocess.Popen,
        output_file: str | Path | None,
        output_encoding: str = "utf-8",
        label: str | None = None,
        cache_output_capture: bool = False,
    ) -> t.Self:
        """Create a new Process from an existing Popen object.

        Currently, only Popen objects with commands specified as a sequence of strings are supported.

        Parameters:
            proc (Popen object): An existing Popen process.
            output_file (PathLike): A filepath representing an output file to monitor. The Popen process must already
                be configured to output to this file.
            output_encoding (optional, t.Any valid string encoding): Which encoding to use when reading the output file,
                defaults to UTF-8.
            label (optional, str): A label to assign to this process. Defaults to the command line arguments.
            cache_output_capture (optional, bool): Whether to cache lines read from the output file.

        Returns:
            Process: A new Process object representing the already-running Popen process. It is recommended that
                you do not use the Popen object reference after it has been passed to this function.
        """
        if hasattr(proc, "environ"):
            env: dict[str, str] = proc.environ()
        else:
            env = psutil.Process(proc.pid).environ()

        if not isinstance(proc.args, (list, tuple)) or not all(isinstance(arg, str) for arg in proc.args):
            raise ValueError("Process.from_existing currently only supports sequences of strings as process commands.")

        new = cls(
            cmd=proc.args,  # type: ignore[arg-type]
            output_file=output_file,
            output_encoding=output_encoding,
            label=label,
            env=env,
            cache_output_capture=cache_output_capture,
        )

        # Store the actual psutil.Process in the new Process object
        new._proc = proc
        # Set the start time using the time reported by the OS
        if hasattr(proc, "create_time"):
            new._start_time = proc.create_time()
        else:
            new._start_time = psutil.Process(proc.pid).create_time()
        # On Windows, don't allow interruption of external processes we didn't create
        new._can_be_interrupted = psutil.LINUX

        assert new.started, "Process was not started at time of creation, or finished and had a null returncode."
        return new

    @classmethod
    def from_pid(
        cls,
        pid: int,
        output_file: str | Path | None,
        output_encoding: str = "utf-8",
        label: str | None = None,
        cache_output_capture: bool = False,
    ) -> t.Self:
        """Create a new Process from the process with the specified PID.

        Parameters:
            pid (int): The process id of a currently-running process. This process MUST be a child of the current
                process unless UNSAFE_ALLOW_NONCHILDREN_CREATION is set.
            output_file (PathLike): A filepath representing an output file to monitor. The Popen process must already
                be configured to output to this file.
            output_encoding (optional, t.Any valid string encoding): Which encoding to use when reading the output file,
                defaults to UTF-8.
            label (optional, str): A label to assign to this process. Defaults to the command line arguments.
            cache_output_capture (optional, bool): Whether to cache lines read from the output file.

        Returns:
            Process: A new Process object representing the already-running process.
        """
        proc = psutil.Process(pid)
        if not _core.UNSAFEALLOW_NONCHILDREN_CREATION and not _is_pid_child_of_current(pid):
            raise _core.UnsafeOperationError(
                f"Cannot create process for pid {pid} which is not a child of the current process."
            )

        with proc.oneshot():
            proc_env = proc.environ()
            args = proc.cmdline()
            start_time = proc.create_time()

        new = cls(
            cmd=args,
            output_file=output_file,
            output_encoding=output_encoding,
            label=label,
            env=proc_env,
            cache_output_capture=cache_output_capture,
        )
        # On Windows, don't allow interruption of external processes we didn't create
        new._can_be_interrupted = psutil.LINUX
        # Set the start time using the time reported by the OS
        new._start_time = start_time
        # Store the actual psutil.Process in the new Process object
        new._proc = proc
        return new

    @property
    def pid(self) -> int | None:
        """The Process ID of this Process (if it has started) otherwise None."""
        return self._proc.pid if self._proc is not None else None

    @property
    def returncode(self) -> int | None:
        """The returncode from this process.

        Returns:
            int | None: The returncode, or None if the process has not yet finished.
        """
        if self._proc is None:
            return None
        return self._poll()

    @cachetools.func.ttl_cache(ttl=POLL_CACHE_TTL)
    def _poll(self):
        # required because psutil.Process doesn't implement ".poll()""
        try:
            return self._proc.poll()
        except AttributeError:
            try:
                return self._proc.wait(timeout=0.01)
            except psutil.TimeoutExpired:
                return None

    @property
    def started(self) -> bool:
        """Whether or not this Process has been started."""
        return self._proc is not None

    @property
    def finished(self) -> bool:
        """Whether or not this Process has finished."""
        return self.returncode is not None

    @property
    def running(self) -> bool:
        """Whether or not this Process is currently running."""
        return self.started and not self.finished

    @property
    def time_since_start(self) -> float | None:
        """The number of seconds since this Process started running, or None if it hasn't been started."""
        if self._start_time is None:
            return None
        return time.time() - self._start_time

    @property
    def can_be_started(self) -> bool:
        """Whether or not this Process can be started."""
        return not self.started

    @property
    def can_be_interrupted(self) -> bool:
        """Whether or not this Process can be interrupted."""
        return self._can_be_interrupted

    @property
    def cmd(self) -> list[str]:
        """A copy of the command and args used for this Process."""
        return self._cmd.copy()

    @cmd.setter
    def cmd(self, cmd: Iterable[str]):
        """Set the command and args used for this Process.

        If the Process has already been started, these cannot be changed.
        """
        if self.started:
            raise _core.UnsafeOperationError("Cannot change command line for an already-started process.")
        self._cmd = list(cmd)

    @property
    def env(self) -> Mapping[str, str] | None:
        """A copy of the environment variables used for this Process."""
        return self._env.copy() if self._env is not None else None

    @env.setter
    def env(self, user_env: Mapping[str, str] | None):
        """Set the environment variables used for this Process.

        If the Process has already been started, these cannot be changed. A value of None means that the current
        process's environment will be used instead.
        """
        if self.started:
            raise _core.UnsafeOperationError("Cannot change environment variables for an already-started process.")
        self._env = _marshall_env_dict(user_env) if user_env is not None else None

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
        f = self.open_output()
        if f is None:
            return []
        try:
            buf = self._maybe_update_buffer(nlines, f)
        finally:
            if not self.cache_output_capture:
                # Storing the output in the buffer only to clear it is a bit inefficient with copying large files
                # but it makes the code clearner.
                self.close_output()
        return buf

    def _maybe_update_buffer(self, nlines: int | None, f: io.TextIOBase):
        """Maybe update the internal linebuffer (if configured) and then return a copy of nlines or all of them)."""
        new = f.readlines()

        if self.cache_output_capture:
            buf = self._output_buffer
            buf.extend(new)
        else:
            buf = new

        return buf[-nlines:] if nlines is not None else buf.copy()

    def monitor(self, nlines: int | None = 10) -> Iterator[list[str]]:
        """Get an iterator that yields the last nlines from the output_file, and completes when the Process is finished.

        Parameters:
            nlines (int): number of lines of output to show.

        Returns:
            list[str]: A list of strings representing the last nlines of output from this Process.
        """
        while self.started:
            output = self.peek_output(nlines)
            if self.finished:
                yield output
                return
            else:
                yield output

    @property
    def state(self) -> t.Literal["complete", "error", "running"] | None:
        """Return a 'state' string or None.

        When the return value is a string, it is compatible with the streamlit Status widget's 'state' setting.
        """
        if self.finished:
            if self.returncode == 0:
                return "complete"
            else:
                return "error"
        elif self.started:
            return "running"
        else:
            return None

    def open_output(self, encoding=None) -> io.TextIOWrapper | None:
        """Open the output read filehandle of this Process object with the specified encoding.

        If the file cannot be found, return None (since the running Process may simply not have created it just yet).
        """
        # we want to keep this filehandle open until closed, so pylint: disable=consider-using-with
        if self.output_file is None:
            return None
        try:
            if self._output_filehandle is None:
                self._output_filehandle = self.output_file.open("r", encoding=encoding or self.output_encoding)
            return self._output_filehandle
        except FileNotFoundError:
            return None

    def close_output(self):
        """Close the output read filehandle of this Process object, if it exists, and clear the internal linebuffer."""
        if self._output_filehandle is not None:
            self._output_filehandle.close()
            self._output_filehandle = None
            # clear the output buffer so we don't get duplicate entries if the file is opened again.
            self._output_buffer.clear()

    def _get_output_destinations(self, filehandle):
        """Return a tuple of STDOUT/STDERR destinations based on the value of filehandle and capture_settings."""
        if filehandle is None:
            return (subprocess.DEVNULL, subprocess.DEVNULL)
        return tuple(filehandle if output_type else subprocess.DEVNULL for output_type in self._capture_settings)

    def __del__(self):
        """Handle garbage collection of this Process."""
        # For now it is safer to interrupt the running process when this object is destroyed since it's
        # far more likely that someone wrote their code wrong and might be producing another process that
        # overlaps with this one. We may change this behavior later
        # TODO: don't terminate if we created our Process from an existing process?
        self.interrupt(wait_for_death=True, force=True)
        self.close_output()


class RerunableProcess(Process):
    """Represents a process which can be re-executed after it has finished."""

    def start(self):
        """Start the process, or restart it if it has finished."""
        if self.finished:
            self._reset()
        super().start()

    def _reset(self):
        """Reset the process after it has finished."""
        self._proc = None
        self._start_time = None
        self._output_buffer = []
        self.close_output()

    @property
    def can_be_started(self) -> bool:
        """Whether the process can be started."""
        return not self.running

    @classmethod
    def from_existing(cls, *_args, **kwargs):  # type: ignore[override]
        """Raise a NotImplementedError."""
        raise NotImplementedError(  # pragma: no cover
            "from_existing() should not be used with RerunnableProcess as there is no reliable "
            "way to ensure the process is re-run with the same inputs. "
            "Use Process.from_existing instead."
        )


class FinalizedProcess(Process):
    """Represents a process which has completed when we weren't watching it, or couldn't be found."""

    def __init__(self, *args, finalized_because: str, returncode: int | None = None, **kwargs):
        """Create a FinalizedProcess with a specific reason and returncode."""
        self._started: t.Literal[True] = False  # type: ignore  # FinalizedProcesses get marked as "started" after init
        try:
            super().__init__(*args, **kwargs)
        finally:
            self._returncode = returncode
            self.finalized_because = finalized_because
            self._started = True

    def start(self):
        """Raise ChildProcessError, FinalizedProcesses cannot be started."""
        raise ChildProcessError("t.Finalized Processes cannot be started")

    @property
    def returncode(self) -> int | None:
        """The returncode of the process."""
        return self._returncode

    @property
    def finished(self) -> t.Literal[True]:
        """Return always True."""
        return True

    @property
    def started(self) -> t.Literal[True]:
        """Return always True."""
        return self._started

    @property
    def can_be_started(self) -> t.Literal[False]:
        """Return always False."""
        return False


def _default_label_if_unset(label: str | None, proc: Process) -> str:
    """Return a concatenated and truncated string from the process's "cmd" if 'label' is None.

    Otherwise return the label.
    """
    if label is None:
        cmd_str = " ".join(proc.cmd).replace("\n", " ")
        return cmd_str[:30] + ("..." if len(cmd_str) > 30 else "")
    return label


def _is_pid_child_of_current(pid: int) -> bool:
    """Check if the provided PID is a child of the currently-running process."""
    current_process = psutil.Process()
    return pid in (descendant.pid for descendant in current_process.children(recursive=True))


def _marshall_env_dict(envdict: Mapping[str, T]) -> dict[str, T]:
    """In windows, environment variables are case insensitive.

    They get returned by psutil and os with uppercase keys always, so here we marshall the keys to uppercase as well.
    """
    if psutil.WINDOWS:
        return {k.upper(): v for k, v in envdict.items()}
    else:
        return dict(**envdict)