"""Core functions and constants for Streamlit Process Manager.

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
