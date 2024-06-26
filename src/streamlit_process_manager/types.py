"""Typechecking imports for Streamlit Process Manager.

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

# pylint: disable=unused-import
from typing import Union as _Union
from streamlit_process_manager.process import Process
from streamlit_process_manager.proxy import ProcessProxy
from streamlit_process_manager.manager import ProcessManager
from streamlit_process_manager.monitor import ProcessMonitor, ProcessMonitorGroup
from streamlit_process_manager.group import ProcessGroup

ProcessOrProxy = _Union[Process, ProcessProxy]

try:
    from typing import Self, TypeAlias  # type: ignore
except ImportError:
    from typing_extensions import Self, TypeAlias  # type: ignore
