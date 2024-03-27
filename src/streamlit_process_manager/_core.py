import typing as t
from pathlib import Path

class UnsafeOperationError(Exception):
    """Error type used when an unsafe process operation is attempted."""


UNSAFE_ALLOW_NONCHILDREN_CREATION = False
"When false (default), prevent creation of Processes from pid's that are not children of the current process."

UNSAFE_ALLOW_NONCHILDREN_TERMINATION = False
"When false (default), prevent termination of Processes with pid's that are not children of the current process."

DT_ONESHOT: t.Final = -1.0
"Indicates that a process monitor should run in 'oneshot' mode and not loop."

DEFAULT_PROCESS_MANAGER_CACHE_PATH: Path = Path(".temp/processmanagercache")
"Default location to store processmanager cachefile."

PROCESS_MANAGER_SESSION_STATE_KEY = "$$$PROCESS_MANAGER$$$"


# Labels for control buttons
INTERRUPT_BTN_LABEL = "🟥 Interrupt"
TERMINATE_BTN_LABEL = "💀 Terminate"
REMOVE_BTN_LABEL = "✖&ensp;Remove Process"
START_BTN_LABEL = "🟢 Start"
RESTART_BTN_LABEL = "🔁 Restart"
