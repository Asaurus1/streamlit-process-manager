"""Streamlit Process Manager User API functions.

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

import os
import typing as t
from pathlib import Path
from collections.abc import Sequence

import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx

from streamlit_process_manager.process import Process, RerunableProcess
from streamlit_process_manager.proxy import ProcessProxy
from streamlit_process_manager.manager import ProcessManager
from streamlit_process_manager.monitor import ProcessMonitor, ProcessMonitorGroup
from streamlit_process_manager import _core

if t.TYPE_CHECKING:
    from streamlit_process_manager.types import ProcessOrProxy
    from _typeshed import StrPath

    ProcessMonitorT = t.TypeVar("ProcessMonitorT", bound=ProcessMonitor)


@st.cache_resource
def get_manager(cachefile: "str | Path | None" = None) -> ProcessManager:
    """Get a new process manager object, or one that already existed stored in streamlit global state."""
    if cachefile is None:
        cachefile = _core.DEFAULT_PROCESS_MANAGER_CACHE_PATH
        if not _core.DEFAULT_PROCESS_MANAGER_CACHE_PATH.parent.exists():
            os.makedirs(_core.DEFAULT_PROCESS_MANAGER_CACHE_PATH.parent, exist_ok=True)
    return ProcessManager(cachefile)


def get_session_manager(cachefile: "str | Path | None" = None) -> ProcessManager:
    """Get a new process manager object, or one that already existed stored in streamlit.session_state."""
    if cachefile is None:
        cachefile = _core.DEFAULT_PROCESS_MANAGER_CACHE_PATH
        if not _core.DEFAULT_PROCESS_MANAGER_CACHE_PATH.parent.exists():
            os.makedirs(_core.DEFAULT_PROCESS_MANAGER_CACHE_PATH.parent, exist_ok=True)
    if _core.PROCESS_MANAGER_SESSION_STATE_KEY not in st.session_state:
        session_id = ctx.session_id if (ctx := get_script_run_ctx()) is not None else "local"
        session_postfix = f".session-{session_id}"
        st.session_state[_core.PROCESS_MANAGER_SESSION_STATE_KEY] = ProcessManager(str(cachefile) + session_postfix)
    return st.session_state[_core.PROCESS_MANAGER_SESSION_STATE_KEY]


@t.overload
def st_process_monitor(
    process: ProcessOrProxy,
    label: "str | None" = None,
    expanded: bool = True,
    lang: str = "log",
    showcontrols: bool = True,
    showruntime: bool = True,
    showlinenumbers: bool = False,
    stripempty: bool = True,
    cls: t.Type[ProcessMonitorT] = ProcessMonitor,  # type: ignore[assignment]
) -> ProcessMonitorT:  # noqa: D103
    ...  # pragma: no cover


@t.overload
def st_process_monitor(
    process: "Sequence[ProcessOrProxy]",
    label: "Sequence[str] | None" = None,
    expanded: bool = False,
    lang: str = "log",
    showcontrols: bool = True,
    showruntime: bool = True,
    showlinenumbers: bool = False,
    stripempty: bool = True,
    cls: t.Type[ProcessMonitorT] = ProcessMonitor,  # type: ignore[assignment]
) -> ProcessMonitorGroup:  # noqa: D103
    ...  # pragma: no cover


def st_process_monitor(
    process,
    label=None,
    expanded=None,
    lang="log",
    showcontrols=True,
    showruntime=True,
    showlinenumbers=False,
    stripempty=True,
    cls: "t.Type[ProcessMonitorT] | None" = None,
) -> ProcessMonitorT | ProcessMonitorGroup:
    """Display a Streamlit process monitor widget for one or more Processes.

    Create a process using the Process object from this module, then pass it to this function to get a ProcessMonitor
    object which you can use to monitor the process output. In it's simplest form, it looks something like this:

        >>> import projects.utils.subprocess_manager as spm
        >>>
        >>> proc = spm.Process(["python", "my_code.py"], output_file="my_code.output").start()
        >>> spm.st_process_monitor(proc).loop_until_finished()

    This code starts a child python process to run "my_code.py" and directs the STDOUT of that process to
    "my_code.output", then displays a process monitor object which tails the contents of "my_code.output"
    until the python process completes.

    You can tune the refresh cycle time as well as the number of lines displayed by the monitor widget:

        >>> import projects.utils.subprocess_manager as spm
        >>>
        >>> proc = spm.Process(["python", "my_code.py"], output_file="my_code.output").start()
        >>> spm.st_process_monitor(proc).loop_until_finished(dt=0.5, nlines=30)
        >>> # Displays the last 30 lines of output and refreshes every 0.5 seconds.

    If your child process writes directly to "my_code.output" that you want to monitor, rather than to STDOUT, you can
    achieve a similar effect with:

        >>> import projects.utils.subprocess_manager as spm
        >>>
        >>> proc = spm.Process(["python", "my_code.py"], output_file="my_code.output", capture="none").start()
        >>> spm.st_process_monitor(proc).loop_until_finished()

    You can start monitor multiple processes at the same time:

        >>> import projects.utils.subprocess_manager as spm
        >>>
        >>> procs = [
        >>>    spm.Process(["python", "my_code.py"], output_file="my_code.output").start(),
        >>>    spm.Process(["python", "my_other_code.py"], output_file="my_other_code.output").start(),
        >>> ]
        >>> spm.st_process_monitor(procs).loop_until_finished()

    To add additonal elements to the process monitor while it's running, you can use the `.loop()` iterator
    instead of `.loop_until_finished()`:

        >>> import projects.utils.subprocess_manager as spm
        >>>
        >>> proc = spm.Process(["python", "my_code.py"], output_file="my_code.output").start()
        >>> for procmon, output in spm.st_process_monitor(proc).loop():
        >>>    if any("Error" in line for line in output):
        >>>       procmon.contents.error("An error occurred")
    """
    _pm_return_value: ProcessMonitorT | ProcessMonitorGroup

    _pm_class: t.Type[ProcessMonitorT] = ProcessMonitor if cls is None else cls  # type: ignore[assignment]

    if isinstance(process, (Process, ProcessProxy)) or not hasattr(process, "__len__"):
        if label is not None and not isinstance(label, str):
            raise TypeError("Provided label must be a single string when a single process is provided.")
        _pm_return_value = _pm_class(
            process=process,
            label=label,
            expanded=expanded if expanded is not None else showcontrols,
            lang=lang,
            show_controls=showcontrols,
            show_line_numbers=showlinenumbers,
            show_runtime=showruntime,
            strip_empty_lines=stripempty,
        )
    else:
        process_seq = process
        del process  # use the plural variable beyond this point
        if label is None:
            _labels = [None] * len(process_seq)
        elif not isinstance(label, str) and len(label) == len(process_seq):
            _labels = list(label)
        else:
            raise ValueError("The number of labels and processes provided must match.")

        _pm_return_value = ProcessMonitorGroup(
            _pm_class(
                process=_proc,
                label=_label,
                expanded=expanded if expanded is not None else False,
                lang=lang,
                show_controls=showcontrols,
                show_line_numbers=showlinenumbers,
                show_runtime=showruntime,
                strip_empty_lines=stripempty,
            )
            for _proc, _label in zip(process_seq, _labels)
        )

    return _pm_return_value


_unspecified = object()  # sentinel


def run(
    args: "Sequence[str]",
    *,
    output_file: "StrPath | None" = _unspecified,  # type: ignore
    encoding: str = "utf-8",
    env: "t.Mapping[str, str] | None" = None,
    cwd: "StrPath | None",
    capture_output: "bool | t.Literal['none', 'stderr', 'stdout', 'all']" = True,
    cache_output_capture: bool = True,
    group: "str | None" = None,
    label: "str | None" = None,
    expanded: bool = False,
    lang: str = "log",
    loop: bool = True,
    nlines: "int | None" = 10,
    dt: float = 1.0,
    showcontrols: bool = True,
    showruntime: bool = True,
    showlinenumbers: bool = False,
    stripempty: bool = True,
    rerunable: bool = False,
    start_trigger: bool = _unspecified,  # type: ignore
) -> ProcessMonitor:
    """Run a subprocess and display its output in a streamlit widget.

        >>> import projects.utils.subprocess_manager as spm
        >>>
        >>> spm.run(["python", "my_code.py"], output_file="my_code.output", loop=True)

    This starts a child python process to run "my_code.py" and directs the STDOUT of that process to
    "my_code.output", then displays a process monitor object which tails the contents of "my_code.output"
    until the process completes.

    The keyword arguments for `run()` are designed to be generally compatible with the `subprocess.run()`
    in the most basic use-cases. The following arguments are supported:
    `args`, `encoding`, `env`, `cwd`, `capture_output`. Other arguments like `check`, `shell, `input`
    `stderr/stdout`, and `timeout` are not supported due to the fact that you'll be running this process
    in the background on streamlit, rather than waiting for it to finish (in fact, in this regard it is
    more similar to subprocess.Popen). However, many of their effects can be replicated using not to much
    additional code.

    This means you can take a program you would normally run with

        >>> subprocess.run(["my_program", "arg1", "arg2"], env={...}, cwd={...})

    and run it in streamlit with

        >>> spm.run(["my_program", "arg1", "arg2"], env={...}, cwd={...})

    `run()` returns a ProcessMonitor object, which is useful if you've set `loop` to false. If you want
    to access the returncode or status of the underlying process you can do the following:

        >>> process_monitor = spm.run(["my_program", "arg1", "arg2"], env={...}, cwd={...})
        >>> process = process_monitor.process
        >>> if process.finished and process.returncode != 0:
        >>>    raise SomeException("Whoops the process failed!")

    Parameters:
        args (Iterable[str]): A sequence of strings representing the command to be run. Shell process commands
            are NOT supported due to the complexities of terminating shell processes.
        output_file (StrPath): A filepath representing an output file to monitor. The Popen process must already
            be configured to output to this file.
        output_encoding (optional, t.Any valid string encoding): Which encoding to use when reading the output file,
            defaults to UTF-8.
        capture_output (bool, or one of "none", "stderr", "stdout", "all"): Which output streams, if t.Any,
            to redirect to output_file in the created process. True=="all", False=="none". Defaults to "all".
        env (optional, Mapping[str, str]): A mapping/dict representing the environment to send to the current process.
            Defaults to the current process's environment if not specified.
        cwd (optional, StrPath): A Path to the directory where this command will be run. Defaults to os.getcwd().
        label (optional, str): A label to assign to this process. Defaults to the command line arguments.
        cache_output_capture (optional, bool): Whether to cache lines read from the output file. Defaults to True.

        group (str): The name of a group to add the process to.
            - If the named group already contains a process, then the one passed via the "process" argument is
            discared.
            - If the named group already contains more than one process, a ValueError is raised.
            - If you are running in a Streamlit environment, then the group name is optional and the default value
            is the ID of the current Streamlit session.

        expanded (optional, bool): Whether to expand the monitor widget while the process is running. Default False.
        lang (optional, str): The language to format process output in within the monitor widget. Defaults to "log".
        loop (optional, bool): If True, `run()` will block execution of the rest of the script until the running
            process finishes. If False, `run()` will not block, the process will run in the background, and the caller
            is responsible for calling `.update()` or `.loop()` on the returned ProcessMonitor object to update the
            monitor widget periodically.
        nlines (optional, int): The max number of output lines to show at a time; set to None to show all lines.
        dt (optional, float): The number of seconds to sleep between each update when `loop=True`.

        showcontrols, showruntime, showlinenumbers (optional, bool): Enable/disable different display aspects
            of the monitoring widget.
        stripempty (optional, bool): If True, the monitor widget will not show any entirely-empty lines
            output from the process. Default True.

        rerunnable (optional, bool): If True, the created Process object will be re-runnable, meaning you can
            call `.start()` on it again once it has finished and it will rerun the same command.
        start_trigger (optional, bool): The Process will attempt to be started every time `run()` is called
            with this value being True. If the Process is already started, nothing will happen.
            Use as a trigger to start the process (e.g. pass in the output of an `st.button` widget or something).
            If unspecified, the process will only be started once, if it has not been started before.

    """
    # TODO: add a 'cwd' argument
    # pylint: disable=too-many-arguments, too-many-locals

    # Get the global ProcessManager
    pm = get_manager()

    if output_file is _unspecified:
        argshash = hash((args, output_file, encoding, env))
        output_file = Path(f".process-manager/process-{argshash}.output")

    if capture_output is True:
        _capture: t.Literal["none", "stderr", "stdout", "all"] = "all"
    elif capture_output is False:
        _capture = "none"
    else:
        _capture = capture_output

    # Configure the Process and start it
    process_cls = RerunableProcess if rerunable else Process
    process = pm.single(
        process_cls(
            args,
            output_file,
            output_encoding=encoding,
            capture=_capture,
            env=env,
            label=label,
            cwd=cwd,
            cache_output_capture=cache_output_capture,
        ),
        group=group,
    )

    if start_trigger is _unspecified:
        start_trigger = not process.started
    if start_trigger:
        process.start_safe()

    # Run the process and monitor it in streamlit!
    pmon: ProcessMonitor = st_process_monitor(
        process=process,
        label=label,
        expanded=expanded,
        lang=lang,
        showcontrols=showcontrols,
        showruntime=showruntime,
        showlinenumbers=showlinenumbers,
        stripempty=stripempty,
    )
    if loop:
        pmon.loop_until_finished(nlines=nlines, dt=dt)

    return pmon
