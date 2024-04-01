"""Streamlit Process Manager User API functions."""

from __future__ import annotations

import os
import typing as t
from collections.abc import Sequence

import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx

from streamlit_process_manager.process import Process, RerunableProcess
from streamlit_process_manager.proxy import ProcessProxy
from streamlit_process_manager.manager import ProcessManager
from streamlit_process_manager.monitor import ProcessMonitor, ProcessMonitorGroup
from streamlit_process_manager import _core

if t.TYPE_CHECKING:
    from pathlib import Path
    from streamlit_process_manager.types import ProcessOrProxy


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
) -> ProcessMonitor:  # noqa: D103
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
) -> ProcessMonitor | ProcessMonitorGroup:
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
    _pm_return_value: ProcessMonitor | ProcessMonitorGroup

    if isinstance(process, (Process, ProcessProxy)) or not hasattr(process, "__len__"):
        if label is not None and not isinstance(label, str):
            raise TypeError("Provided label must be a single string when a single process is provided.")
        _pm_return_value = ProcessMonitor(
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
            ProcessMonitor(
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
    args,
    *,
    output_file=_unspecified,
    encoding="utf-8",
    env=None,
    group=None,
    label=None,
    expanded=None,
    lang="log",
    loop=True,
    nlines=10,
    dt=1.0,
    showcontrols=True,
    showruntime=True,
    showlinenumbers=False,
    stripempty=True,
    rerunable=False,
    start_trigger=_unspecified,
):
    """Run a subprocess and display its output in a streamlit widget.

        >>> import projects.utils.subprocess_manager as spm
        >>>
        >>> spm.run(["python", "my_code.py"], output_file="my_code.output", loop=True)

    This starts a child python process to run "my_code.py" and directs the STDOUT of that process to
    "my_code.output", then displays a process monitor object which tails the contents of "my_code.output"
    until the process completes.
    """
    # TODO: add a 'cwd' argument
    # pylint: disable=too-many-arguments

    # Get the global ProcessManager
    pm = get_manager()

    if output_file is _unspecified:
        argshash = hash((args, output_file, encoding, env))
        output_file = Path(f".process-manager/process-{argshash}.output")

    # Configure the Process and start it
    process_cls = RerunableProcess if rerunable else Process
    process = pm.single(
        process_cls(args, output_file, output_encoding=encoding, env=env, label=label),
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
