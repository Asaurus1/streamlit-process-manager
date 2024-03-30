"""Typechecking imports for Streamlit Process Manager."""
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
