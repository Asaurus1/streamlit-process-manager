from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from streamlit_process_manager.process import Process
    from streamlit_process_manager.manager import ProcessManager
    from streamlit_process_manager.monitor import ProcessMonitor, ProcessMonitorGroup
    from streamlit_process_manager.group import ProcessGroup

    try:
        from typing import Self, TypeAlias  # type: ignore
    except ImportError:
        from typing_extensions import Self, TypeAlias  # type: ignore
