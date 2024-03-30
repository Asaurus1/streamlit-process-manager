from contextlib import contextmanager
import time
import typing as t
from collections.abc import Iterable, Iterator, Sequence
from datetime import timedelta

import streamlit as st
from streamlit.elements.lib.mutable_status_container import StatusContainer
from streamlit.runtime.scriptrunner import get_script_run_ctx

from streamlit_process_manager.process import Process
from streamlit_process_manager.proxy import ProcessProxy
from streamlit_process_manager._core import (
    INTERRUPT_BTN_LABEL,
    TERMINATE_BTN_LABEL,
    REMOVE_BTN_LABEL,
    START_BTN_LABEL,
    RESTART_BTN_LABEL,
)

if t.TYPE_CHECKING:
    from streamlit_process_manager.types import Self, TypeAlias

    ProcessOrProxy: "TypeAlias" = "Process | ProcessProxy"
else:
    ProcessOrProxy = None


stu = None  # temp

class ProcessMonitor:
    """Represents a streamlit widget which shows the current status and controls for a single Process."""

    class PMConfig(t.NamedTuple):
        """Read-only configuration settings for a ProcessMonitor."""

        label: str
        show_controls: bool
        show_runtime: bool
        show_line_numbers: bool
        strip_empty_lines: bool
        output_language: str

    def __init__(
        self,
        process: ProcessOrProxy,
        label: "str | None" = None,
        expanded: "bool | None" = None,
        lang: str = "log",
        show_controls: bool = True,
        show_runtime: bool = True,
        show_line_numbers: bool = False,
        strip_empty_lines: bool = True,
    ):
        """Create a monitor widget for a single Process."""
        self.process: ProcessOrProxy = process
        self.config = self.PMConfig(
            label=label or process.label,
            show_controls=show_controls,
            show_runtime=show_runtime,
            show_line_numbers=show_line_numbers,
            strip_empty_lines=strip_empty_lines,
            output_language=lang,
        )

        # Render
        self.status: StatusContainer = st.status(**self._eval_status_state(expanded=expanded))  # type: ignore
        self._controls_empty: "st._DeltaGenerator" = self.status.empty()
        self._output_emtpy: "st._DeltaGenerator" = self.status.empty()
        self.contents: "st._DeltaGenerator" = self.status.container()  # A place for "loop" callers to put content
        self._draw_controls()

    def update(self, nlines: "int | None" = 10) -> "t.List[str]":
        """Update the output display and status widget state for this ProcessMonitor.

        Parameters:
            nlines (int): number of lines of output to show.

        Returns:
            t.List[str]: A list of strings representing nlines from each monitor updated this cycle
        """
        lines = []
        if (
            self.process.started
            and self.process.output_file is not None
            and self.config.output_language.lower() != "off"
        ):
            lines = self._get_lines_from_process(nlines)
            self._output_emtpy.code(
                "".join(lines), language=self.config.output_language, line_numbers=self.config.show_line_numbers
            )
        self.status.update(**self._eval_status_state())  # type: ignore
        return lines

    def loop(self, nlines: "int | None" = 10, dt: float = 1.0) -> "Iterable[t.Tuple[Self, t.List[str]]]":
        """Get a generator which yields (self, output as t.List[str]) from a running process.

        Parameters:
            nlines (int): number of lines of output to show.
            dt (float): minimum amount of time to pause between loops.

        Yields:
            self: this ProcessMonitorGroup object
            output (t.List[str]): A list of strings representing nlines from each monitor updated this cycle
        """
        proc = self.process
        was_running = False
        while True:
            output = self.update(nlines=nlines)
            was_running = proc.running
            yield_start = time.monotonic()
            yield (self, output)

            if dt >= 0 and proc.running:
                yield_dt = time.monotonic() - yield_start
                # st_sleep checks in with streamlit every 0.3 seconds to see if it wants to rerun.
                _st_sleep(max(0, dt - yield_dt))

            # Note: in theory this conditional should be was_running != proc.running, since we would also want to update
            # the button state in the event that the user starts the process between loop iterations.
            # In practice, this is unlikely, however, and fewer force-reruns is better. The user can always call their
            # own rerun after they start the process, and when the start happens because of a button press, a rerun
            # is implicit.
            if self.config.show_controls and was_running is True and proc.running is False:
                # If the process just stopped, the control buttons will now be in an unexpected state.
                # We can't run _draw_controls() again because streamlit will throw a DuplicateWidgetID exception.
                # Instead, we request a rerun and hope that the user has cached things appropriately.
                st.rerun()

            # If process is not running by this point, then break out of the loop
            if not proc.running:
                break

        proc.close_output()  # close the reader after we're done

    def loop_until_finished(self, nlines: "int | None" = 10, dt: float = 1.0) -> "Self":
        """Block execution and `loop` until the monitored Process is no longer running.

        Returns: this ProcessMonitor
        """
        for _ in self.loop(nlines=nlines, dt=dt):
            pass
        return self

    def _draw_controls(self):
        """Draw the control buttons for this process, if configured to do so."""
        if self.config.show_controls:
            with self._controls_empty.container():
                _render_process_control_buttons(self.process)

    def _eval_status_state(self, expanded: "bool | None" = None) -> "t.Dict[str, str | bool]":
        """Determine the keyword args to pass to `st.status.update()` this loop, based on process status."""
        proc = self.process
        escaped_label = self.config.label.replace("]", "&#93;").replace("[", "&#91;")
        status_kwds: t.Dict[str, t.Any] = {"label": escaped_label}

        if (process_state := proc.state) is not None:
            status_kwds["state"] = process_state

        if (rc := proc.returncode) not in (0, None):
            status_kwds["label"] += f" :red[(finished with errorcode: {rc})]"
        elif not proc.started:
            status_kwds["label"] += " :gray[(not started)]"
        elif proc.running and self.config.show_runtime and (runtime := proc.time_since_start) is not None:
            status_kwds["label"] += f" :gray[(running for {_runtime_format(runtime)})]"

        if expanded is not None:
            status_kwds["expanded"] = expanded

        return status_kwds

    def _get_lines_from_process(self, nlines: "int | None") -> "t.List[str]":
        lines = self.process.peek_output(nlines=nlines)
        if self.config.strip_empty_lines:
            return [aline for aline in lines if aline.strip()]
        return lines


class ProcessMonitorGroup(Sequence):  # type: Sequence[ProcessMonitor]
    """Represents a group of ProcessMonitors."""

    def __init__(self, monitors: "Iterable[ProcessMonitor]"):
        """Create a group of ProcessMonitors from an iterable."""
        self._monitors: t.List[ProcessMonitor] = list(monitors)

    def update(self, nlines: "int | None" = 10) -> "t.List[t.List[str]]":
        """Run a single `update` cycle for ever monitor in this group.

        Parameters:
            nlines (int): number of lines of output to show.

        Returns:
            t.List[t.List[str]]: A list of list of strings representing nlines from each monitor updated this cycle
        """
        return [monitor.update(nlines=nlines) for monitor in self._monitors]

    def loop(self, nlines: "int | None" = 10, dt: float = 1.0) -> "Iterator[t.Tuple[Self, t.List[t.List[str]]]]":
        """Get a generator which yields (self, output as t.List[t.List[str]]) from a running process.

        Parameters:
            nlines (int): number of lines of output to show.
            dt (float): minimum amount of time to pause between loops.

        Yields:
            self: this ProcessMonitorGroup object
            output (t.List[t.List[str]]): A list of list of strings representing nlines from each monitor updated this cycle
        """
        last_running_states = running_states = [False] * len(self._monitors)
        while True:
            output_list = self.update(nlines=nlines)
            last_running_states = [monitor.process.running for monitor in self._monitors]
            yield_start = time.monotonic()
            yield self, output_list
            running_states = [monitor.process.running for monitor in self._monitors]

            if dt >= 0 and any(running_states):
                yield_dt = time.monotonic() - yield_start
                # st_sleep checks in with streamlit every 0.3 seconds to see if it wants to rerun.
                _st_sleep(max(0, dt - yield_dt))
                running_states = [monitor.process.running for monitor in self._monitors]

            # Note: in theory this conditional should be was_running != proc.running, since we would also want to update
            # the button state in the event that the user starts the process between loop iterations.
            # In practice, this is unlikely, however, and fewer force-reruns is better. The user can always call their
            # own rerun after they start the process, and when the start happens because of a button press, a rerun
            # is implicit.
            if any(
                monitor.config.show_controls and was_running is True and is_running is False
                for monitor, is_running, was_running in zip(self._monitors, running_states, last_running_states)
            ):
                # If t.Any process has stopped during the sleep, the control buttons will now be in an unexpected state.
                # We can't run _draw_controls() again because streamlit will throw a DuplicateWidgetID exception.
                # Instead, we request a rerun and hope that the user has cached things appropriately.
                # If ALL processes have stopped, then the next rerun should skip this step so we don't get into
                # an infinite loop.
                st.rerun()

            # If no processes are running by this point, then break out of the loop
            if not any(running_states):
                break

        for monitor in self._monitors:
            monitor.process.close_output()  # close the readers after we're done

    def loop_until_finished(self, nlines: "int | None" = 10, dt: float = 1.0) -> "Self":
        """Block execution and update all ProcessMonitors until their Processes are no longer running.

        Parameters:
            nlines (int): number of lines of output to show.
            dt (float): minimum amount of time to pause between loops.

        Returns: this ProcessMonitorGroup
        """
        for _ in self.loop(nlines=nlines, dt=dt):
            pass
        return self

    def __iter__(self) -> "Iterator[ProcessMonitor]":
        """Iterate over the ProcessMonitors in this group."""
        return iter(self._monitors)

    def __len__(self) -> int:
        """Get the number of ProcessMonitors in this group."""
        return len(self._monitors)

    @t.overload
    def __getitem__(self, index: int) -> ProcessMonitor:  # noqa: D105
        ...  # pragma: no cover

    @t.overload
    def __getitem__(self, _slice: slice) -> "t.List[ProcessMonitor]":  # noqa: D105
        ...  # pragma: no cover

    def __getitem__(self, index):
        """Get ProcessMonitor(s) from the specified index or slice."""
        return self._monitors[index]


def _render_process_control_buttons(process: ProcessOrProxy):
    """Draws a set of control streamlit buttons for the specified Process side-by-side.

    If 'remove_cb' is given, a "Remove Process" button will be displayed, with
    an 'on_click' event set to the callback and the Process itself as the first and only argument.
    """
    cols = st.columns((25, 25, 40, 20))

    # All callbacks are wrapped in "_wrap_exception" so that they kindly display t.Any exception messages encountered
    # when called (such as UnsafeOperationErrors)

    # Start/restart button
    start_label = RESTART_BTN_LABEL if process.finished else START_BTN_LABEL
    cols[0].button(
        start_label,
        disabled=not process.can_be_started,
        on_click=_wrap_exception()(process.start_safe),  # type: ignore  # streamlit doesn't like that we return a value
        key=f"startbutton: {process.pid or id(process)}",
    )

    # Interrupt button
    interrupt_label = INTERRUPT_BTN_LABEL if process.can_be_interrupted else TERMINATE_BTN_LABEL
    cols[1].button(
        interrupt_label,
        disabled=not process.running,
        on_click=_wrap_exception()(process.interrupt),
        kwargs=dict(force=True),
        key=f"interruptbutton: {process.pid or id(process)}",
    )

    # Remove button (if Process or proxy supports removal)
    if hasattr(process, "supports_remove") and t.cast(ProcessProxy, process).supports_remove:
        cols[2].button(
            REMOVE_BTN_LABEL,
            key=f"removebutton: {process.pid or id(process)}",
            help="Clear this process from the group",
            on_click=_wrap_exception()(t.cast(ProcessProxy, process).remove_from_pgroup),
            disabled=process.running,
        )


def _runtime_format(process_runtime: float) -> str:
    """Format a process runtime into a readable format."""
    return str(timedelta(seconds=round(process_runtime)))


@contextmanager
def _wrap_exception():
    """Wrap a function so that any exceptions are caught and displayed in streamlit, but your page continues to run."""
    try:
        yield
    except Exception as exc:
        st.exception(exc)
    finally:
        st.divider()


def _st_yield():
    """Check if the streamlit process wants to handle a re-run request.

    Call in the middle of a loop to allow your long process to terminate quickly.
    """
    # pylint: disable=protected-access
    try:
        get_script_run_ctx().session_state._yield_callback()
    except AttributeError:
        pass


def _st_sleep(time_sec: float):
    """Sleep for a specified amount of time, yielding periodically."""
    start = time.time()
    time_remaining = start + time_sec - time.time()
    while time_remaining > 0:
        time.sleep(min(time_remaining, 0.3))
        _st_yield()
        time_remaining = start + time_sec - time.time()
