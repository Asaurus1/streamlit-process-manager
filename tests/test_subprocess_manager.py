import copy
import io
import os
import signal
import subprocess
import sys
import time
import unittest.mock as mock
from pathlib import Path
import typing as t

import psutil
import pytest
import streamlit.testing.v1.app_test

from streamlit_process_manager import api, group, proxy, process, manager, monitor, _core

# Because we're testing an internal module here, we pylint: disable=protected-access
# And because pytest fixtured, we pylint: disable=redefined-outer-name, unused-argument


# Test Fixtures -----------------------------------------------------
@pytest.fixture(scope="session")
def TEST_OUTPUT_PATH():
    rand = str(hash(time.time() + id(pytest)))[-5:]
    path = str(Path(f"tests/test_data/spm{rand}.output"))
    yield path
    if os.path.exists(path):
        delete_file_with_timeout(path)


@pytest.fixture(scope="session")
def TEST_PROCESS_MANAGER_PATH():
    rand = str(hash(time.time() + id(pytest)))[-5:]
    path = str(Path(f"tests/test_data/process_monitor_cache{rand}"))
    yield path
    if os.path.exists(path):
        delete_file_with_timeout(path)


@pytest.fixture
def with_mock_popen():
    with mock.patch("psutil.Popen", new=mock_subprocess):
        yield


@pytest.fixture
def fake_process(TEST_OUTPUT_PATH):
    proc = process.Process(["foo", "bar"], TEST_OUTPUT_PATH)
    proc._raise_if_proc_not_child = lambda *args, **kwargs: True  # because we don't have a real process attached
    yield proc
    proc.close_output()
    proc._proc = None  # unset _proc at the end so that __del__ doesn't throw exceptions


@pytest.fixture
def fake_rerunable_process(TEST_OUTPUT_PATH):
    proc = process.RerunableProcess(["foo", "bar"], TEST_OUTPUT_PATH)
    proc._raise_if_proc_not_child = lambda *args, **kwargs: True  # because we don't have a real process attached
    yield proc
    proc.close_output()
    proc._proc = None  # unset _proc at the end so that __del__ doesn't throw exceptions


@pytest.fixture
def fake_process_seq(fake_process, fake_rerunable_process):
    return [fake_process, fake_rerunable_process]


@pytest.fixture
def started_test_process_fh(fake_process: process.Process):
    start_with_mock_sp(fake_process)
    with open(fake_process.output_file, "w", encoding="utf-8", buffering=1) as f:
        yield f


@pytest.fixture
def real_process_short(TEST_OUTPUT_PATH):
    proc = process.Process([sys.executable, "tests/test_data/subprocess_loop.py", "5"], TEST_OUTPUT_PATH)
    yield proc
    proc.close_output()
    proc.terminate(wait_for_death=True)


@pytest.fixture
def real_process_3s(TEST_OUTPUT_PATH):
    proc = process.Process([sys.executable, "tests/test_data/subprocess_loop.py", "30"], TEST_OUTPUT_PATH)
    psutil.Popen([sys.executable, "tests/test_data/subprocess_loop.py", "30"])
    yield proc
    proc.close_output()
    proc.terminate(wait_for_death=True)


@pytest.fixture
def real_process_infinite(TEST_OUTPUT_PATH):
    proc = process.Process([sys.executable, "tests/test_data/subprocess_loop.py", "10000000000"], TEST_OUTPUT_PATH)
    yield proc
    proc.close_output()
    proc.terminate(wait_for_death=True)


@pytest.fixture
def get_manager(request, TEST_PROCESS_MANAGER_PATH):
    """A fixture which wraps 'api.get_manager' so that any managers that get created through it automatically
    get cleaned up at the end of the test.
    """

    def _get():
        manager = api.get_manager(TEST_PROCESS_MANAGER_PATH)

        def _safe_deconstruct():
            try:
                manager.__del__()
            except _core.UnsafeOperationError:
                pass

        request.addfinalizer(_safe_deconstruct)
        return manager

    return _get


@pytest.fixture
def p_manager(get_manager):
    return get_manager()


@pytest.fixture
def pretend_windows():
    with mock.patch("psutil.WINDOWS", new=True),\
        mock.patch("psutil.LINUX", new=False),\
        mock.patch("subprocess.CREATE_NEW_PROCESS_GROUP", new=0x200, create=True),\
        mock.patch("signal.CTRL_BREAK_EVENT", new=1, create=True)\
    :
        yield


@pytest.fixture
def pretend_linux():
    with mock.patch("psutil.WINDOWS", new=False),\
        mock.patch("psutil.LINUX", new=True)\
    :
        yield


# Test Helpers -----------------------------------------------------
def start_with_mock_sp(proc: process.Process, *args, **kwargs):
    with mock.patch("psutil.Popen", wraps=mock_subprocess) as mock_sp:
        proc.start(*args, **kwargs)
    return mock_sp


def mark_as_finished(proc, rc=0):
    if isinstance(proc, process.Process):
        mark_as_finished(proc._proc, rc=rc)
        proc._poll.cache_clear()
    else:
        proc.returncode = rc


def mock_subprocess(cmd, *args, **kwargs):
    spec = dir(psutil.Process)
    # spec.remove("poll")  # remove because psutil.Process doesn't have "poll"
    mock_sp = mock.MagicMock(spec=spec)

    def _poll(*args, **kwargs):
        return mock_sp.returncode

    mock_sp.args = cmd
    mock_sp.returncode = None
    mock_sp.terminate.side_effect = lambda: mark_as_finished(mock_sp, -1)
    mock_sp.send_signal.side_effect = lambda *args: mark_as_finished(mock_sp, -2)
    mock_sp.wait.side_effect = _poll
    mock_sp.pid = 123456
    mock_sp.create_time.return_value = time.time()

    return mock_sp


def delete_file_with_timeout(file, *, timeout=3.0):
    start = time.time()
    last_exc = None
    while time.time() < start + timeout:
        try:
            os.remove(file)
            return
        except (PermissionError, OSError) as exc:
            last_exc = exc
            time.sleep(0.5)
    raise last_exc


def app_loop_until_finished():
    # pylint: disable=import-outside-toplevel
    import streamlit as st

    from streamlit_process_manager.monitor import ProcessMonitor

    proc = st.session_state["proc"]
    procmonargs = st.session_state["procmonargs"]

    ProcessMonitor(proc, **procmonargs).loop_until_finished()


def app_monitor_update_once():
    # pylint: disable=import-outside-toplevel
    import streamlit as st

    from streamlit_process_manager.monitor import ProcessMonitor

    proc = st.session_state["proc"]
    procmonargs = st.session_state["procmonargs"]

    ProcessMonitor(proc, **procmonargs).update(nlines=20)


def app_monitor_func():
    # pylint: disable=import-outside-toplevel
    import streamlit as st

    from streamlit_process_manager import st_process_monitor

    procs = st.session_state["procs"]
    procmonargs = st.session_state["procmonargs"]

    st.session_state.monitor_out = st_process_monitor(procs, **procmonargs).update(nlines=20)


# Process Tests -----------------------------------------------------
def test_process_sets_label_correctly():
    new_process = process.Process(["a", "b", "c"], output_file="test")
    assert new_process.label == "a b c"

    new_process = process.Process(["a", "b", "c"], output_file="test", label="custom")
    assert new_process.label == "custom"


def test_process_capture_settings():
    new_process = process.Process(["a", "b", "c"], output_file="test", capture="none")
    assert new_process._capture_settings == (False, False)
    assert new_process._get_output_destinations("foo") == (subprocess.DEVNULL, subprocess.DEVNULL)

    new_process = process.Process(["a", "b", "c"], output_file="test", capture="stderr")
    assert new_process._capture_settings == (False, True)
    assert new_process._get_output_destinations("foo") == (subprocess.DEVNULL, "foo")

    new_process = process.Process(["a", "b", "c"], output_file="test", capture="stdout")
    assert new_process._capture_settings == (True, False)
    assert new_process._get_output_destinations("foo") == ("foo", subprocess.DEVNULL)

    new_process = process.Process(["a", "b", "c"], output_file="test", capture="all")
    assert new_process._capture_settings == (True, True)
    assert new_process._get_output_destinations("foo") == ("foo", "foo")


def test_process_capture_setting_error():
    with pytest.raises(ValueError, match="Invalid capture setting 'doofenshmirtz'"):
        process.Process(["a"], output_file=None, capture="doofenshmirtz")

    with pytest.raises(ValueError, match="Cannot request output capture without setting an output_file"):
        process.Process(["a"], output_file=None, capture="all")


def test_process_started(fake_process: process.Process):
    assert not fake_process.started
    start_with_mock_sp(fake_process)
    assert fake_process.started


def test_process_start_safe_fails():
    new_process = process.FinalizedProcess(
        ["foo", "bar"], output_file=None, capture="none", finalized_because="this is a test", returncode=0
    )
    with mock.patch.object(new_process, "start") as mock_start:
        assert new_process.start_safe() is new_process
        mock_start.assert_not_called()


def test_set_cmd_env_for_started_process_fails():
    new_process = process.FinalizedProcess(
        ["foo", "bar"], output_file=None, capture="none", finalized_because="this is a test", returncode=0
    )
    with pytest.raises(_core.UnsafeOperationError):
        new_process.cmd = ["baz"]

    with pytest.raises(_core.UnsafeOperationError):
        new_process.env = {"hello": "world"}


def test_real_process_infinite_works(real_process_infinite: process.Process):
    real_process_infinite.start()
    time.sleep(0.5)
    assert real_process_infinite.running  # make sure it hasn't crashed


def test_process_finished(fake_process: process.Process):
    assert not fake_process.finished
    start_with_mock_sp(fake_process)
    assert not fake_process.finished
    fake_process.terminate()
    assert fake_process.finished


def test_process_running(fake_process: process.Process):
    assert not fake_process.running
    start_with_mock_sp(fake_process)
    assert fake_process.running
    fake_process.terminate()
    assert not fake_process.running


def test_process_state(fake_process: process.Process):
    assert fake_process.state is None
    start_with_mock_sp(fake_process)
    assert fake_process.state == "running"
    mark_as_finished(fake_process, rc=0)
    assert fake_process.state == "complete"
    mark_as_finished(fake_process, rc=5)
    assert fake_process.state == "error"


def test_process_monitoriter(fake_process: process.Process):
    monitor = fake_process.monitor(5)
    # Raises immediately if not started
    with pytest.raises(StopIteration):
        next(fake_process.monitor())
    # Now start the process
    start_with_mock_sp(fake_process)
    with open(fake_process.output_file, "w", encoding="utf8") as f:
        assert next(monitor) == []
        f.write("Hello\nworld\n")
        f.flush()
        assert next(monitor) == ["Hello\n", "world\n"]
        f.write("a\nb\nc\nd\n")
        f.flush()
        assert next(monitor) == ["world\n", "a\n", "b\n", "c\n", "d\n"]
    mark_as_finished(fake_process)
    # Make sure it emits final output one last time after finishing
    assert next(monitor) == ["world\n", "a\n", "b\n", "c\n", "d\n"]
    # then stops
    with pytest.raises(StopIteration):
        next(monitor)


def test_process_pid(fake_process: process.Process):
    assert fake_process.pid is None
    start_with_mock_sp(fake_process)
    assert fake_process.pid == 123456


def test_process_terminate_twice(real_process_infinite: process.Process):
    real_process_infinite.start()
    assert real_process_infinite.started
    real_process_infinite.terminate(wait_for_death=True)
    real_process_infinite.terminate()


def test_process_open_close(fake_process: process.Process, TEST_OUTPUT_PATH):
    assert fake_process._output_filehandle is None
    start_with_mock_sp(fake_process)  # must start in order to create the file
    assert os.path.isfile(TEST_OUTPUT_PATH)
    assert isinstance(fake_process.open_output(), io.TextIOWrapper)
    assert fake_process._output_filehandle.name == TEST_OUTPUT_PATH
    fake_process.close_output()
    assert fake_process._output_filehandle is None


def test_process_open_fails_on_nonextant_file(fake_process: process.Process):
    fake_process.output_file = Path("tests/test_data/some_other_test_file")
    assert not os.path.isfile(fake_process.output_file), "File exists, cannot run test!"
    assert fake_process.open_output() is None
    assert fake_process._output_filehandle is None


def test_process_open_output_with_no_file(fake_process: process.Process):
    fake_process.output_file = None
    start_with_mock_sp(fake_process)
    assert fake_process.open_output() is None
    assert fake_process.peek_output() == []
    assert fake_process._output_filehandle is None


def test_process_start_common_args(fake_process: process.Process):
    mock_sp = start_with_mock_sp(fake_process)
    assert mock_sp.call_args[0][0] == fake_process.cmd
    assert mock_sp.call_args[1]["env"] == fake_process.env


def test_process_start_with_no_file(fake_process: process.Process):
    fake_process.output_file = None
    mock_sp = start_with_mock_sp(fake_process)
    assert mock_sp.call_args[1]["stdout"] == subprocess.DEVNULL
    assert mock_sp.call_args[1]["stderr"] == subprocess.DEVNULL
    assert fake_process._output_filehandle is None


def test_process_start_with_file(fake_process: process.Process):
    mock_sp = start_with_mock_sp(fake_process)
    assert Path(mock_sp.call_args[1]["stdout"].name) == fake_process.output_file
    assert mock_sp.call_args[1]["stdout"].mode == "w"
    assert mock_sp.call_args[1]["stderr"] is mock_sp.call_args[1]["stdout"]


def test_started_creation_flags_windows(fake_process: process.Process, pretend_windows):
    mock_sp = start_with_mock_sp(fake_process)
    # Currently disabled; see Process.interrupt() for more info
    # assert mock_sp.call_args[1]["creationflags"] == subprocess.CREATE_NEW_PROCESS_GROUP
    assert mock_sp.call_args[1]["creationflags"] == 0


def test_started_creation_flags_linux(fake_process: process.Process, pretend_linux):
    mock_sp = start_with_mock_sp(fake_process)
    assert mock_sp.call_args[1]["creationflags"] == 0


def test_process__del__():
    new_process = process.Process(["a", "b", "c"], output_file="test")
    patch_close = new_process.close_output = mock.MagicMock()
    patch_interrupt = new_process.interrupt = mock.MagicMock()
    new_process.__del__()
    patch_close.assert_called_once()
    patch_interrupt.assert_called_once_with(wait_for_death=True, force=True)


def test_process_time_since_start(fake_process: process.Process):
    assert fake_process.time_since_start is None
    start_with_mock_sp(fake_process)
    time.sleep(0.1)
    assert fake_process.time_since_start < time.time()


def test_process_buffers_output(fake_process: process.Process, started_test_process_fh: io.TextIOWrapper):
    assert fake_process.cache_output_capture is True
    assert len(fake_process._output_buffer) == 0
    started_test_process_fh.write("foo\nbar\n")
    fake_process.peek_output()
    assert len(fake_process._output_buffer) == 2
    started_test_process_fh.write("baz\n")
    fake_process.peek_output()
    assert len(fake_process._output_buffer) == 3


def test_process_peek_output(fake_process: process.Process, started_test_process_fh: io.TextIOWrapper):
    started_test_process_fh.write("foo\nbar\n")
    assert fake_process.peek_output() == ["foo\n", "bar\n"]
    started_test_process_fh.write("baz\n" * 10)
    assert fake_process.peek_output(nlines=1) == ["baz\n"]
    assert fake_process.peek_output(nlines=None) == ["foo\n", "bar\n"] + ["baz\n"] * 10


def test_process_peek_output_with_no_file(fake_process: process.Process, started_test_process_fh: io.TextIOWrapper):
    started_test_process_fh.write("foo\n")
    with mock.patch.object(fake_process, "open_output", return_value=None):
        assert fake_process.peek_output() == []


def test_process_does_not_buffer_when_asked(fake_process: process.Process, started_test_process_fh: io.TextIOWrapper):
    fake_process.cache_output_capture = False
    assert len(fake_process._output_buffer) == 0
    started_test_process_fh.write("foo\nbar\n")
    assert fake_process.peek_output() == ["foo\n", "bar\n"]
    assert len(fake_process._output_buffer) == 0


def test_process_cannot_be_restarted(fake_process: process.Process):
    start_with_mock_sp(fake_process)
    fake_process.terminate()
    with pytest.raises(ChildProcessError):
        start_with_mock_sp(fake_process)


def test_process_cannot_be_started_while_running(fake_process: process.Process):
    start_with_mock_sp(fake_process)
    with pytest.raises(ChildProcessError):
        start_with_mock_sp(fake_process)


def test_rerunable_process_can_be_restarted_only_when_finished(fake_rerunable_process: process.RerunableProcess):
    start_with_mock_sp(fake_rerunable_process)
    with pytest.raises(ChildProcessError):
        start_with_mock_sp(fake_rerunable_process)
    fake_rerunable_process.terminate()
    start_with_mock_sp(fake_rerunable_process)


def test_rerunable_process_reset(fake_rerunable_process: process.RerunableProcess):
    start_with_mock_sp(fake_rerunable_process)
    fake_rerunable_process._reset()
    start_with_mock_sp(fake_rerunable_process)


def test_process_cannot_interrupt_non_child(fake_process: process.Process):
    # the fixture overwrites this attribute in the instance so we have to
    # remove that here if we want the original function to work.
    del fake_process._raise_if_proc_not_child

    start_with_mock_sp(fake_process)
    fake_process._proc.pid = os.getpid()  # current process PID, which is not a child of itself
    with pytest.raises(_core.UnsafeOperationError):
        fake_process.terminate()
        fake_process.interrupt(force=True)

    # this is allowed if we set the global flags
    fake_process._proc.terminate.reset_mock()
    with mock.patch.object(_core, "UNSAFE_ALLOW_NONCHILDREN_TERMINATION", new=True):
        fake_process.terminate()
        fake_process.interrupt(force=True)
    fake_process._proc.terminate.assert_called()


def test_cannot_create_process_from_non_child_pid():
    current_process_pid = os.getpid()
    with pytest.raises(_core.UnsafeOperationError, match="not a child of the current process"):
        proc = process.Process.from_pid(current_process_pid, output_file="test")

    # this is allowed if we set the global flags
    try:
        with mock.patch.object(_core, "UNSAFE_ALLOW_NONCHILDREN_CREATION", new=True):
            proc = process.Process.from_pid(current_process_pid, output_file="test")
        assert proc.pid == current_process_pid
    finally:
        proc._proc = None  # make sure we don't kill the current process if we raise an exception


@pytest.mark.skip("Windows interrupt currently disabled, see Process.interrupt() for more info.")
def test_process_interrupt_win_allowed(fake_process: process.Process, pretend_windows):
    start_with_mock_sp(fake_process)
    fake_process._can_be_interrupted = True
    fake_process.interrupt(wait_for_death=True, force=True)
    fake_process._proc.send_signal.assert_called_once_with(signal.CTRL_BREAK_EVENT)
    fake_process._proc.wait.assert_called_once()

    fake_process._proc.send_signal.reset_mock()
    fake_process._proc.wait.reset_mock()
    fake_process.interrupt(wait_for_death=False, force=True)
    fake_process._proc.send_signal.assert_called_once_with(signal.CTRL_BREAK_EVENT)
    fake_process._proc.wait.assert_not_called()

    fake_process._proc.send_signal.reset_mock()
    fake_process._proc.wait.reset_mock()
    fake_process.interrupt(wait_for_death=False, force=False)
    fake_process._proc.send_signal.assert_called_once_with(signal.CTRL_BREAK_EVENT)
    fake_process._proc.wait.assert_not_called()


def test_process_interrupt_win_not_allowed(fake_process: process.Process, pretend_windows):
    start_with_mock_sp(fake_process)
    fake_process._can_be_interrupted = False
    fake_process.interrupt(wait_for_death=True, force=True)
    fake_process._proc.terminate.assert_called_once()
    fake_process._proc.wait.assert_called_once()

    fake_process._proc.terminate.reset_mock()
    fake_process._proc.wait.reset_mock()
    fake_process.interrupt(wait_for_death=False, force=True)
    fake_process._proc.terminate.assert_called_once()
    fake_process._proc.wait.assert_not_called()

    fake_process._proc.terminate.reset_mock()
    fake_process._proc.wait.reset_mock()
    with pytest.raises(_core.UnsafeOperationError, match="This process cannot be safely interrupted"):
        fake_process.interrupt(wait_for_death=False, force=False)
    fake_process._proc.send_signal.assert_not_called()
    fake_process._proc.terminate.assert_not_called()
    fake_process._proc.wait.assert_not_called()


def test_process_interrupt_linux(fake_process: process.Process, pretend_linux):
    start_with_mock_sp(fake_process)
    fake_process.interrupt(wait_for_death=True, force=True)
    fake_process._proc.send_signal.assert_called_once_with(signal.SIGINT)
    fake_process._proc.wait.assert_called_once()

    fake_process._proc.send_signal.reset_mock()
    fake_process._proc.wait.reset_mock()
    fake_process.interrupt(wait_for_death=False, force=True)
    fake_process._proc.send_signal.assert_called_once_with(signal.SIGINT)
    fake_process._proc.wait.assert_not_called()

    fake_process._proc.send_signal.reset_mock()
    fake_process._proc.wait.reset_mock()
    fake_process.interrupt(wait_for_death=False, force=False)
    fake_process._proc.send_signal.assert_called_once_with(signal.SIGINT)
    fake_process._proc.wait.assert_not_called()


def test_process_interrupt_terminate_silent_if_no_such_process_linux(fake_process: process.Process, pretend_linux):
    start_with_mock_sp(fake_process)
    fake_process._proc.send_signal.side_effect = psutil.NoSuchProcess("process already dead")
    fake_process.interrupt()
    fake_process.terminate()
    fake_process._proc.send_signal.assert_called()
    fake_process._proc.terminate.assert_called()


def test_process_interrupt_terminate_silent_if_no_such_process_win(fake_process: process.Process, pretend_windows):
    start_with_mock_sp(fake_process)
    fake_process._proc.terminate.side_effect = psutil.NoSuchProcess("process already dead")
    fake_process.interrupt(force=True)
    fake_process.terminate()
    fake_process._proc.terminate.assert_called()


def test_process_to_dict(fake_process: process.Process, TEST_OUTPUT_PATH: str, pretend_windows):
    output = fake_process.to_dict()
    expected = dict(
        pid=None,
        cmd=["foo", "bar"],
        label="foo bar",
        output_file=TEST_OUTPUT_PATH,
        env=None,
        output_encoding="utf-8",
        start_time=None,
        state=None,
        returncode=None,
    )
    assert output == expected

    fake_process.env = {"boo": "far"}
    start_with_mock_sp(fake_process)
    output = fake_process.to_dict()
    expected = dict(
        pid=123456,
        cmd=["foo", "bar"],
        label="foo bar",
        output_file=TEST_OUTPUT_PATH,
        env={"BOO": "far"},
        output_encoding="utf-8",
        start_time=pytest.approx(time.time()),
        state="running",
        returncode=None,
    )
    assert output == expected


def test_process_from_pid(real_process_infinite: process.Process):
    real_process_infinite.env = {"SYSTEMROOT": "C:\Windows", "PYTHONUTF8": "1", "PYTHONHASHSEED": "074"}
    real_process_infinite.start()
    time.sleep(0.1)
    new_process = process.Process.from_pid(real_process_infinite.pid, output_file="test")
    assert not isinstance(new_process, process.FinalizedProcess)
    assert new_process.pid == real_process_infinite.pid
    assert new_process._start_time == pytest.approx(real_process_infinite._start_time, abs=0.1)
    assert process._do_evn_dicts_match(new_process.env, real_process_infinite.env)
    assert new_process.cmd == real_process_infinite.cmd
    assert new_process.output_file == Path("test")


def test_process_from_dict_not_started(TEST_OUTPUT_PATH: str):
    new_process_dict = process.Process.SavedProcessDict(
        pid=None,
        cmd=["foo", "bar"],
        label="baz",
        state=None,
        env={"HELLO": "world"},
        output_file=TEST_OUTPUT_PATH,
        output_encoding="utf-8",
        start_time=170000,
        returncode=None,
    )
    new_process = process.Process.from_dict(new_process_dict)
    assert new_process.pid is None
    assert new_process.running is False
    assert new_process.label == "baz"
    assert new_process.cmd == ["foo", "bar"]
    assert new_process.output_file == Path(TEST_OUTPUT_PATH)
    assert new_process.output_encoding == "utf-8"
    assert new_process.env == {"HELLO": "world"}
    assert new_process.state is None
    assert new_process._proc is None
    assert new_process._start_time is None  # because it's not started


def test_process_from_dict_started(real_process_infinite: process.Process):
    real_process_infinite.env = {"SYSTEMROOT": "C:\Windows", "PYTHONUTF8": "1", "PYTHONHASHSEED": "074"}
    real_process_infinite.start()
    time.sleep(0.1)
    new_process = process.Process.from_dict(real_process_infinite.to_dict())
    assert new_process.pid == real_process_infinite.pid
    assert new_process.running is True
    assert new_process.label == real_process_infinite.label
    assert new_process.cmd == real_process_infinite.cmd
    assert new_process.output_file == real_process_infinite.output_file
    assert new_process.output_encoding == real_process_infinite.output_encoding
    assert process._do_evn_dicts_match(new_process.env, real_process_infinite.env)
    assert new_process.state == "running"
    assert new_process._proc is not None
    assert new_process._start_time == real_process_infinite._start_time


def test_process_proxy_eq(fake_process: process.Process):
    proxy1 = proxy.ProcessProxy(fake_process)
    proxy2 = proxy.ProcessProxy(fake_process)
    assert fake_process == proxy1
    assert fake_process == proxy2
    assert proxy2 == fake_process
    assert proxy1 == fake_process
    assert proxy1 == proxy2
    assert proxy2 == proxy1
    # "break" the weakref on proxy1
    proxy1._proc_weak = lambda: None
    assert proxy1 != proxy2
    assert proxy2 != proxy1
    assert fake_process != proxy1
    assert fake_process == proxy2


def test_finalized_for_nonexsistant(real_process_short: process.Process):
    real_process_short.start()
    real_process_short._proc.wait(timeout=1)
    running_dict = real_process_short.to_dict()
    new_process = process.Process.from_dict(running_dict)
    assert isinstance(new_process, process.FinalizedProcess)
    assert new_process.finalized_because == f"Process with pid {real_process_short.pid} no longer exists."
    assert new_process.returncode == 0


def test_finalize_for_incorrect_start_time(real_process_infinite: process.Process):
    proc = real_process_infinite
    proc.start()
    running_dict = real_process_infinite.to_dict()
    running_dict["start_time"] = 0
    new_process = process.Process.from_dict(running_dict)
    assert isinstance(new_process, process.FinalizedProcess)
    assert (
        new_process.finalized_because == f"Process with pid {real_process_infinite.pid} has an unexpected create time."
    )
    assert new_process.returncode is None


def test_finalized_for_wrong_env(real_process_infinite: process.Process):
    proc = real_process_infinite
    proc.start()
    running_dict = real_process_infinite.to_dict()
    running_dict["env"] = {"WRONG": "env"}
    new_process = process.Process.from_dict(running_dict)
    assert isinstance(new_process, process.FinalizedProcess)
    assert (
        new_process.finalized_because == f"Process with pid {real_process_infinite.pid} has an unexpected environment."
    )
    assert new_process.returncode is None


def test_finalized_hardcoded_props():
    fproc = process.FinalizedProcess(
        ["foo", "bar"], output_file=None, capture="none", returncode=199, finalized_because="the death star"
    )
    assert fproc.returncode == 199
    assert fproc.finalized_because == "the death star"
    assert fproc.finished is True
    assert fproc.started is True
    assert fproc.can_be_started is False
    with pytest.raises(ChildProcessError, match="Finalized Processes cannot be started"):
        fproc.start()


def test_process_from_existing_psutilPopen():
    new_proc = mock_subprocess(["run", "my", "program"])
    new_proc.pid = 123456
    new_proc.environ.return_value = {"thanks": "for", "the": "fish"}

    try:
        new_process = process.Process.from_existing(new_proc, output_file="42.txt")

        assert new_process.cmd == ["run", "my", "program"]
        assert new_process.env == process._marshall_env_dict({"thanks": "for", "the": "fish"})
        assert new_process.pid == 123456
        assert new_process._start_time > 0
    finally:
        new_proc.pid = None  # needed because otherwise our PID is 123456 and we wil throw an exception on __del__


def test_process_from_existing_subprocessPopen():
    new_proc = mock_subprocess(["run", "my", "program"])
    new_proc.pid = 123456
    del new_proc.create_time
    del new_proc.environ

    try:

        with mock.patch("psutil.Process") as process_inspector:
            process_inspector().environ.return_value = {"thanks": "for", "the": "fish"}
            process_inspector().create_time.return_value = time.time()
            new_process = process.Process.from_existing(new_proc, output_file="42.txt")

        assert new_process.cmd == ["run", "my", "program"]
        assert new_process.env == process._marshall_env_dict({"thanks": "for", "the": "fish"})
        assert new_process.pid == 123456
        assert new_process._start_time > 0
    finally:
        new_proc.pid = None  # needed because otherwise our PID is 123456 and we wil throw an exception on __del__


def test_process_from_existing_only_supports_string_seq_commands():
    bad_commands = [
        b"bytes",
        "str",
        Path("myprogram.exe"),
    ]
    for cmd in bad_commands:
        new_proc = mock_subprocess(cmd)
        with pytest.raises(ValueError, match="Process.from_existing currently only supports sequences of strings"):
            process.Process.from_existing(new_proc, output_file=None)


# ProcessGroup Tests -----------------------------------------------------
def test_process_group_cannot_add_same_process_twice(fake_process: process.Process):
    with pytest.raises(ValueError, match="Not all input Processes are unique"):
        pg = group.ProcessGroup([fake_process, fake_process])

    pg = group.ProcessGroup()
    pg.add(fake_process)
    with pytest.raises(ValueError, match="Process already exists in ProcessGroup"):
        pg.add(fake_process)


def test_process_group_empty():
    pg = group.ProcessGroup()
    assert len(pg) == 0
    assert list(pg) == []
    assert pg.is_empty is True
    assert pg.procs == []
    assert pg.returncodes == []


def test_process_group_nonempty(fake_process: process.Process):
    pg = group.ProcessGroup([fake_process])
    assert len(pg) == 1
    assert list(pg) == [fake_process]
    assert pg.is_empty is False
    assert pg.procs == [fake_process]
    assert pg.returncodes == [None]
    assert pg[0] == fake_process
    with pytest.raises(IndexError):
        _ = pg[1]


def test_process_group_start_all(fake_process: process.Process, with_mock_popen):
    pg = group.ProcessGroup([fake_process])
    assert not fake_process.running
    pg.start_all()
    assert fake_process.running


def test_process_group_clear(fake_process: process.Process):
    pg = group.ProcessGroup([fake_process])
    assert len(pg) == 1
    start_with_mock_sp(fake_process)
    pg.unsafe_clear()
    assert not fake_process.finished
    assert len(pg) == 0


def test_process_group_remove(fake_process: process.Process):
    pg = group.ProcessGroup([fake_process])
    assert len(pg) == 1
    start_with_mock_sp(fake_process)
    with pytest.raises(_core.UnsafeOperationError, match="The process .* remove from this group is running"):
        pg.remove(fake_process)
    mark_as_finished(fake_process)
    pg.remove(fake_process)
    assert len(pg) == 0


def test_process_group_pop_finished(fake_process_seq: t.List[process.Process]):
    for test in fake_process_seq:
        start_with_mock_sp(test)
    pg = group.ProcessGroup(fake_process_seq)
    assert pg.pop_finished() == []
    assert len(pg.procs) == 2
    fake_process_seq[0].terminate()
    assert pg.pop_finished() == [fake_process_seq[0]]
    assert len(pg.procs) == 1


def test_process_group_terminate_all(fake_process_seq: t.List[process.Process]):
    for test in fake_process_seq:
        start_with_mock_sp(test)
    pg = group.ProcessGroup(fake_process_seq)
    assert len(pg.procs) == 2
    assert pg.returncodes == [None, None]
    pg.terminate_all()
    assert pg.returncodes == [-1, -1]


def test_process_group_interrupt_all(fake_process_seq: t.List[process.Process], pretend_linux):
    for test in fake_process_seq:
        start_with_mock_sp(test)
    pg = group.ProcessGroup(fake_process_seq)
    assert len(pg.procs) == 2
    assert pg.returncodes == [None, None]
    pg.interrupt_all(force=False)
    assert pg.returncodes == [-2, -2]


def test_process_group_terminate_all_ingnores_exceptions(fake_process_seq: t.List[process.Process]):
    for test in fake_process_seq:
        start_with_mock_sp(test)
    pg = group.ProcessGroup(fake_process_seq)
    fake_process_seq[0]._proc.terminate.side_effect = ValueError("Houston, we've had a problem")
    fake_process_seq[1]._proc.terminate.side_effect = ValueError("Open the door, HAL")
    with pytest.raises(ValueError, match="Open the door, HAL"):
        pg.terminate_all()


def test_process_group_interrupt_all_ingnores_exceptions(fake_process_seq: t.List[process.Process], pretend_linux):
    for test in fake_process_seq:
        start_with_mock_sp(test)
    pg = group.ProcessGroup(fake_process_seq)
    fake_process_seq[0]._proc.send_signal.side_effect = ValueError("Houston, we've had a problem")
    fake_process_seq[1]._proc.send_signal.side_effect = ValueError("Open the door, HAL")
    with pytest.raises(ValueError, match="Open the door, HAL"):
        pg.interrupt_all(force=False)


def test_process_group_labels(fake_process: process.Process):
    fake_process.label = "may_the_force_be_with_you.exe"
    pg = group.ProcessGroup([fake_process])
    assert pg.labels == ["may_the_force_be_with_you.exe"]


def test_process_group_by_label(fake_process: process.Process, fake_rerunable_process: process.Process):
    fake_process.label = "Aang.sh"
    fake_rerunable_process.label = "Katara.bat"
    pg = group.ProcessGroup([fake_process, fake_rerunable_process])
    # Note: by_label actually returns ProcessProxies but their __eq__ allows comparing them to a Process.
    assert pg.by_label("a", match_whole=False) == [fake_process, fake_rerunable_process]
    assert pg.by_label("Aang.sh") == [fake_process]
    assert pg.by_label("foobar") == []
    assert pg.by_label("Kat", match_whole=False) == [fake_rerunable_process]


# ProcessManager Tests -----------------------------------------------------
def test_get_manager(TEST_PROCESS_MANAGER_PATH):
    _manager = api.get_manager(TEST_PROCESS_MANAGER_PATH)
    assert _manager._cachefilehandle.name == TEST_PROCESS_MANAGER_PATH
    assert isinstance(_manager, manager.ProcessManager)

    _manager = api.get_manager(Path(TEST_PROCESS_MANAGER_PATH))
    assert _manager._cachefilehandle.name == TEST_PROCESS_MANAGER_PATH


@pytest.mark.skip("this one is just really annoying to test right now")
def test_get_manager_calls_osmakedirs_on_default_path(TEST_PROCESS_MANAGER_PATH): ...  # noqa: E704


def test_get_manager_caching_app(TEST_PROCESS_MANAGER_PATH):

    def _get_manager_script():
        # pylint: disable=import-outside-toplevel, reimported
        import streamlit as st

        import streamlit_process_manager as spm

        pm = spm.get_manager(st.session_state.managerpath)
        st.session_state.this_run_manager = pm

    try:
        at = streamlit.testing.v1.AppTest.from_function(_get_manager_script)
        at.session_state.managerpath = TEST_PROCESS_MANAGER_PATH

        at.run(timeout=1)
        first_run_mgr = at.session_state.this_run_manager
        assert isinstance(first_run_mgr, manager.ProcessManager)

        at.run(timeout=1)
        second_run_mgr = at.session_state.this_run_manager
        assert first_run_mgr is second_run_mgr

        # Clear the cache
        api.get_manager.clear()
        at.run(timeout=1)
        third_run_mgr = at.session_state.this_run_manager
        assert first_run_mgr is not third_run_mgr
    finally:
        # Remove all references to the managers at the end so that they close themselves.
        api.get_manager.clear()
        del at.session_state.this_run_manager


def test_manager_init(p_manager: manager.ProcessManager):
    assert p_manager.groups == []
    assert p_manager.to_dict() == {}


def test_manager_group_creation(p_manager: manager.ProcessManager):
    assert isinstance(p_manager.group("foo"), group.ProcessGroup)
    assert len(p_manager.groups) == 1


def test_manager_groups_returns_copy(p_manager: manager.ProcessManager):
    assert p_manager.groups is not p_manager._groups


def test_manager_add_process_one(fake_process: process.Process, p_manager: manager.ProcessManager, with_mock_popen):
    p_manager.add(fake_process, "test_group", start=False)
    assert fake_process.started is False
    assert p_manager.group("test_group")[0] == fake_process
    assert all(isinstance(_proxy, proxy.ProcessProxy) for _proxy in p_manager.group("test_group"))
    p_manager.add(fake_process, "test_group_2", start=True)
    assert fake_process.started is True


def test_manager_add_process_many(fake_process: process.Process, p_manager: manager.ProcessManager, with_mock_popen):
    test_process2 = copy.copy(fake_process)
    p_manager.add([fake_process, test_process2], "test_group", start=False)
    assert fake_process.started is False
    assert test_process2.started is False
    assert p_manager.group("test_group")[0] == fake_process
    assert p_manager.group("test_group")[1] == test_process2
    assert all(isinstance(_proxy, proxy.ProcessProxy) for _proxy in p_manager.group("test_group"))
    p_manager.add([fake_process, test_process2], "test_group2", start=True)
    assert fake_process.started is True
    assert test_process2.started is True


def test_manager_single(fake_process: process.Process, p_manager: manager.ProcessManager, with_mock_popen):
    p_manager.single(fake_process, "single_group", start=False)
    assert fake_process.started is False
    assert p_manager.group("single_group")[0] == fake_process
    assert isinstance(p_manager.group("single_group")[0], proxy.ProcessProxy)
    p_manager.single(fake_process, "single_group_2", start=True)
    assert fake_process.started is True


def test_manager_single_nogroup(fake_process: process.Process, p_manager: manager.ProcessManager):
    with pytest.raises(RuntimeError, match="Cannot create single process without streamlit session context"):
        p_manager.single(fake_process)

    with mock.patch("streamlit.runtime.scriptrunner.get_script_run_ctx") as mock_get_ctx:
        mock_get_ctx().session_id = "owhatagooseiam"
        p_manager.single(fake_process)
        assert "single_group_for_session_owhatagooseiam" in p_manager.groups


def test_manager_single_returns_existing(fake_process: process.Process, p_manager: manager.ProcessManager):
    p_manager.single(fake_process, "single_group")
    assert p_manager.single("fake_value", "single_group") == fake_process
    assert isinstance(p_manager.single("fake_value", "single_group"), proxy.ProcessProxy)


def test_manager_single_wont_add_existing_group(fake_process: process.Process, p_manager: manager.ProcessManager):
    test_process2 = copy.copy(fake_process)
    p_manager.add([fake_process, test_process2], "test_group")
    with pytest.raises(
        ValueError,
        match="Cannot create/get single process for group .* as the group already exists "
        r"and has more than one process in it\.",
    ):
        p_manager.single(fake_process, "test_group")


def test_manager_add_wont_cover_single(fake_process: process.Process, p_manager: manager.ProcessManager):
    p_manager.single(fake_process, group="test_single")
    p_manager.group("test_single").unsafe_clear()
    assert len(p_manager.group("test_single")) == 0

    # even after a single group is cleared, disallow adds
    assert "test_single" in p_manager._single_groups
    with pytest.raises(ValueError, match=r"test_single was previously populated using the .single\(\) method."):
        p_manager.add(fake_process, group="test_single")


def test_manager_to_dict(fake_process: process.Process, p_manager: manager.ProcessManager):
    p_manager.single(fake_process, "single_group")
    p_manager.single(fake_process, "single_group2")
    to_dict = p_manager.to_dict()
    assert to_dict.keys() == {"single_group", "single_group2"}
    assert all(isinstance(v, list) for v in to_dict.values())

    to_dict = p_manager.to_dict(["single_group"])
    assert to_dict.keys() == {"single_group"}
    assert all(isinstance(v, list) for v in to_dict.values())

    to_dict = p_manager.to_dict(["bad_key"])
    assert to_dict.keys() == set()


def test_manager_read_write(fake_process: process.Process, get_manager):
    first_manager: manager.ProcessManager = get_manager()
    first_manager.add(fake_process, "test_group")
    first_manager._cachefilehandle.close()  # close the filehandle before we create a new manager that opens it.
    new_manager: manager.ProcessManager = get_manager()
    new_manager._read_from_disk()

    assert first_manager._groups.keys() == new_manager._groups.keys()
    for group in first_manager._groups:
        for proc1, proc2 in zip(first_manager.group(group), new_manager.group(group)):
            assert proc1.cmd == proc2.cmd
            assert proc1.pid == proc2.pid
            assert proc1.env == proc2.env
            assert proc1._start_time == proc2._start_time


def test_manager_read_bad_yaml(p_manager: manager.ProcessManager):
    with mock.patch("json.load", return_value=["not", "a", "dict"]):
        with pytest.raises(ValueError, match="Bad cache data"):
            p_manager._read_from_disk()


# ProcessMonitor tests ---------------------------------------------------------
@mock.patch.object(process.Process, "peek_output")
def test_process_monitor_core(mock_output, fake_process: process.Process):
    pm = monitor.ProcessMonitor(fake_process)

    assert pm._get_lines_from_process(10) == []
    assert pm._get_lines_from_process(10) == []
    assert pm._eval_status_state() == dict(label="foo bar :gray[(not started)]")
    start_with_mock_sp(fake_process)
    mock_output.return_value = ["test\n", "lines\n"]
    assert pm._eval_status_state() == dict(label="foo bar :gray[(running for 0:00:00)]", state="running")
    assert pm._get_lines_from_process(10) == ["test\n", "lines\n"]
    time.sleep(0.5)
    assert pm._eval_status_state() == dict(label="foo bar :gray[(running for 0:00:01)]", state="running")
    mark_as_finished(fake_process, rc=0)
    assert pm._eval_status_state() == dict(label="foo bar", state="complete")
    mark_as_finished(fake_process, rc=5)
    assert pm._eval_status_state() == dict(label="foo bar :red[(finished with errorcode: 5)]", state="error")
    assert pm._get_lines_from_process(10) == ["test\n", "lines\n"]


@mock.patch.object(process.Process, "peek_output")
def test_process_monitor_skip_empty_lines(mock_output, fake_process: process.Process):
    pm = monitor.ProcessMonitor(fake_process)
    start_with_mock_sp(fake_process)
    mock_output.return_value = ["test\n", "\n", "\n", "lines\n"]
    assert pm._get_lines_from_process(10) == ["test\n", "lines\n"]
    pm = monitor.ProcessMonitor(fake_process, strip_empty_lines=False)
    assert pm._get_lines_from_process(10) == ["test\n", "\n", "\n", "lines\n"]


def test_process_monitor_update(fake_process: process.Process):
    pm = monitor.ProcessMonitor(fake_process)

    assert pm._get_lines_from_process(10) == []


def test_process_monitor_group_update(fake_process_seq: t.List[process.Process]):
    pmg = monitor.ProcessMonitorGroup(monitor.ProcessMonitor(proc) for proc in fake_process_seq)

    with mock.patch.object(monitor.ProcessMonitor, "update") as mock_update:
        pmg.update()
        assert mock_update.call_count == len(fake_process_seq)


def test_process_monitor_group_dunders(fake_process_seq: t.List[process.Process]):
    pmg = monitor.ProcessMonitorGroup(monitor.ProcessMonitor(proc) for proc in fake_process_seq)
    assert len(pmg) == len(fake_process_seq)
    assert [pm.process for pm in pmg] == fake_process_seq
    assert pmg[0].process == fake_process_seq[0]


@mock.patch.object(monitor, "st")
def test_process_monitor_group_loop(mock_streamlit, fake_process_seq: t.List[process.Process]):
    pmg = monitor.ProcessMonitorGroup(monitor.ProcessMonitor(proc) for proc in fake_process_seq)
    assert pmg.loop_until_finished() is pmg
    loop_iter = pmg.loop(dt=0)

    selfref, output_list = next(loop_iter)
    assert selfref is pmg
    assert output_list == [[], []]
    mock_streamlit.status().update.assert_called_with(label="foo bar :gray[(not started)]")

    mock_streamlit.status().update.reset_mock()
    start_with_mock_sp(fake_process_seq[0])
    selfref, output_list = next(loop_iter)
    assert selfref is pmg
    assert output_list == [[], []]
    proc_start_time = monitor._runtime_format(fake_process_seq[0].time_since_start)
    mock_streamlit.status().update.assert_has_calls(
        [
            mock.call(label=f"foo bar :gray[(running for {proc_start_time})]", state="running"),
            mock.call(label="foo bar :gray[(not started)]"),
        ]
    )

    mock_streamlit.status().update.reset_mock()
    mark_as_finished(fake_process_seq[0])
    mock_streamlit.rerun.side_effect = RuntimeError("Rerun Exception")
    with pytest.raises(RuntimeError, match="Rerun Exception"):
        selfref, output_list = next(loop_iter)
    mock_streamlit.rerun.assert_called_once()
    # This simulates a "rerun" where .loop() would be called again fresh.
    loop_iter = pmg.loop(dt=0)
    selfref, output_list = next(loop_iter)
    assert selfref is pmg
    assert output_list == [[], []]
    mock_streamlit.status().update.assert_has_calls(
        [
            mock.call(label="foo bar", state="complete"),
            mock.call(label="foo bar :gray[(not started)]"),
        ]
    )


def test_process_monitor_app(real_process_infinite: process.Process):

    at = streamlit.testing.v1.AppTest.from_function(app_monitor_update_once)
    at.session_state["proc"] = real_process_infinite
    at.session_state["procmonargs"] = dict(show_controls=True, label="foo bar")
    at.run(timeout=6)

    # Before process is started
    assert at.status[0].label == "foo bar :gray[(not started)]"
    assert at.status[0].state == "running"

    start_btn = at.status[0].button[0]
    assert start_btn.label == "🟢 Start"
    assert start_btn.disabled is False

    terminate_interrupt_btn = at.status[0].button[1]
    assert terminate_interrupt_btn.label in (_core.INTERRUPT_BTN_LABEL, _core.TERMINATE_BTN_LABEL)
    assert terminate_interrupt_btn.disabled is True

    assert len(at.status[0].code) == 0

    # While process running
    real_process_infinite.start()
    at.run(timeout=6)

    assert (
        at.status[0].label
        == f"foo bar :gray[(running for {monitor._runtime_format(real_process_infinite.time_since_start)})]"
    )
    assert at.status[0].state == "running"

    start_btn = at.status[0].button[0]
    assert start_btn.label == _core.START_BTN_LABEL
    assert start_btn.disabled is True

    terminate_interrupt_btn = at.status[0].button[1]
    assert terminate_interrupt_btn.label in (_core.INTERRUPT_BTN_LABEL, _core.TERMINATE_BTN_LABEL)
    assert terminate_interrupt_btn.disabled is False

    output_code = at.status[0].code[0]
    assert output_code.language == "log"

    # After process finished
    time.sleep(1)
    real_process_infinite.terminate()
    at.run(timeout=6)

    assert at.status[0].label == f"foo bar :red[(finished with errorcode: {real_process_infinite.returncode})]"
    assert at.status[0].state == "error"

    start_btn = at.status[0].button[0]
    assert start_btn.label == _core.RESTART_BTN_LABEL
    assert start_btn.disabled is True

    terminate_interrupt_btn = at.status[0].button[1]
    assert terminate_interrupt_btn.label in (_core.INTERRUPT_BTN_LABEL, _core.TERMINATE_BTN_LABEL)
    assert terminate_interrupt_btn.disabled is True

    output_code = at.status[0].code[0]
    assert output_code.language == "log"
    print(output_code.value)
    assert all(int(i) >= 0 for i in output_code.value.splitlines()), "some output is not an integer as expected"


def test_process_monitor_app_loop(real_process_3s: process.Process):
    at = streamlit.testing.v1.AppTest.from_function(app_loop_until_finished)
    at.session_state["proc"] = real_process_3s.start()
    at.session_state["procmonargs"] = dict(show_controls=True, label="foo bar")
    at.run(timeout=6)

    assert at.status[0].label == "foo bar"
    assert at.status[0].state == "complete"

    start_btn = at.status[0].button[0]
    assert start_btn.label == _core.RESTART_BTN_LABEL
    assert start_btn.disabled is True

    terminate_interrupt_btn = at.status[0].button[1]
    assert terminate_interrupt_btn.label in (_core.INTERRUPT_BTN_LABEL, _core.TERMINATE_BTN_LABEL)
    assert terminate_interrupt_btn.disabled is True

    output_code = at.status[0].code[0]
    assert output_code.language == "log"
    assert output_code.value == "\n".join(str(i) for i in range(20, 30))


def test_process_monitor_app_no_controls(real_process_short: process.Process):
    at = streamlit.testing.v1.AppTest.from_function(app_monitor_update_once)
    at.session_state["proc"] = real_process_short
    at.session_state["procmonargs"] = dict(show_controls=False, label="foo bar")
    at.run(timeout=6)

    assert at.status[0].label == "foo bar :gray[(not started)]"
    assert at.status[0].state == "running"

    assert len(at.status[0].button) == 0


def test_process_monitor_app_no_runtime(real_process_short: process.Process):
    at = streamlit.testing.v1.AppTest.from_function(app_monitor_update_once)
    at.session_state["proc"] = real_process_short.start()
    at.session_state["procmonargs"] = dict(show_runtime=False, label="foo bar")
    at.run(timeout=6)

    assert real_process_short.time_since_start > 0
    assert at.status[0].label == "foo bar"
    assert at.status[0].state == "running"


def test_process_monitor_app_removal_button(real_process_short: process.Process):
    pg = group.ProcessGroup()
    _proxy = pg.add(real_process_short)
    at = streamlit.testing.v1.AppTest.from_function(app_monitor_update_once)
    at.session_state["proc"] = _proxy.start()
    at.session_state["procmonargs"] = dict(label="foo bar")
    at.run(timeout=6)

    remove_btn = at.status[0].button[2]
    assert remove_btn.label == _core.REMOVE_BTN_LABEL
    remove_btn.click()

    real_process_short.interrupt(force=True, wait_for_death=True)

    with mock.patch.object(pg, "remove", wraps=pg.remove) as mock_remove:
        at.run(timeout=6)
        assert len(at.exception) == 0
        mock_remove.assert_called_with(real_process_short)
        assert len(pg) == 0


# st_process_monitor tests -----------------------------------------------------
def test_spm_single_process(fake_process: process.Process):
    pm = api.st_process_monitor(process=fake_process)
    assert isinstance(pm, monitor.ProcessMonitor)


def test_spm_single_expanded_defaults(fake_process: process.Process):
    with mock.patch.object(api, "ProcessMonitor") as mock_pm:
        api.st_process_monitor(process=fake_process, expanded=True)
        assert mock_pm.call_args[1]["expanded"] is True

        api.st_process_monitor(process=fake_process, expanded=False)
        assert mock_pm.call_args[1]["expanded"] is False

        api.st_process_monitor(process=fake_process, showcontrols=True)
        assert mock_pm.call_args[1]["expanded"] is True

        api.st_process_monitor(process=fake_process, showcontrols=False)
        assert mock_pm.call_args[1]["expanded"] is False


def test_spm_bad_labels(fake_process: process.Process, fake_process_seq: t.List[process.Process]):
    with pytest.raises(TypeError, match="Provided label must be a single string when a single process is provided."):
        api.st_process_monitor(process=fake_process, label=["foo", "bar"])
    with pytest.raises(ValueError, match="The number of labels and processes provided must match."):
        api.st_process_monitor(process=fake_process_seq, label="foo")


def test_spm_multi_process(fake_process_seq: t.List[process.Process]):
    pmg = api.st_process_monitor(process=fake_process_seq, label=["one", "two"])
    assert isinstance(pmg, monitor.ProcessMonitorGroup)
    assert all(pm.process == tp for pm, tp in zip(pmg, fake_process_seq))
    assert [pm.config.label for pm in pmg] == ["one", "two"]


def test_spm_multi_expanded_defaults(fake_process_seq: t.List[process.Process]):
    with mock.patch.object(api, "ProcessMonitor") as mock_pm:
        api.st_process_monitor(process=fake_process_seq, expanded=True)
        assert mock_pm.call_args[1]["expanded"] is True

        api.st_process_monitor(process=fake_process_seq, expanded=False)
        assert mock_pm.call_args[1]["expanded"] is False

        api.st_process_monitor(process=fake_process_seq, showcontrols=True)
        assert mock_pm.call_args[1]["expanded"] is False

        api.st_process_monitor(process=fake_process_seq, showcontrols=False)
        assert mock_pm.call_args[1]["expanded"] is False


def test_spm_multi_no_removal_on_non_mutable(fake_process_seq: t.List[process.Process]):
    fake_process_tup = tuple(fake_process_seq)
    pmg = api.st_process_monitor(process=fake_process_tup, label=["one", "two"])
    assert isinstance(pmg, monitor.ProcessMonitorGroup)
    assert all(not getattr(pm.process, "supports_remove", None) for pm in pmg)


def test_process_monitor_func_app(real_process_3s, real_process_short, real_process_infinite):
    at = streamlit.testing.v1.AppTest.from_function(app_monitor_func)
    at.session_state["procs"] = group.ProcessGroup([real_process_short, real_process_3s, real_process_infinite])
    at.session_state["procmonargs"] = dict(label=["p1", "p2", "p3"])
    at.run(timeout=1)  # should not timeout since processes are not started yet
    # TODO: finish test


# Helper Function Tests -----------------------------------------------------
def test_default_label_if_unset(fake_process: process.Process):
    assert process._default_label_if_unset("set_label", fake_process) == "set_label"
    assert process._default_label_if_unset(None, fake_process) == "foo bar"
    fake_process.cmd = ["a"] * 50
    assert process._default_label_if_unset(None, fake_process) == " ".join(["a"] * 15) + " ..."


def test_is_pid_child_of_current(real_process_infinite):
    real_process_infinite.start()
    assert process._is_pid_child_of_current(real_process_infinite.pid)
    assert not process._is_pid_child_of_current(0)


def test_marshall_env_dict_win(pretend_windows):
    assert process._marshall_env_dict({"foo": "bar"}) == {"FOO": "bar"}


def test_marshall_env_dict_linux(pretend_linux):
    assert process._marshall_env_dict({"foo": "bar"}) == {"foo": "bar"}


def test_runtime_format():
    assert monitor._runtime_format(0.5) == "0:00:00"
    assert monitor._runtime_format(100) == "0:01:40"
    assert monitor._runtime_format(5000000) == "57 days, 20:53:20"
