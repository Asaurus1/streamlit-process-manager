"""Example of how to use the subprocess_manager module.

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

import sys
from pathlib import Path

import streamlit as st

# Our examples sometimes pylint: disable=redefined-outer-name

counter_prog = """
import time

try:
    for x in range(20):
        print(f"Running... {x}%")
        time.sleep(0.1)
finally:
    print("I'm done!")
"""

endless_prog = """
import time, sys

while True:
    sys.stdout.truncate(0)
    sys.stdout.seek(0)
    print("This is a process that never ends\\n\\n")
    time.sleep(1)
    print("It just goes on and on my friend\\n\\n")
    time.sleep(1)
    print("Some people started runnin' it")
    time.sleep(1)
    print("Not knowing what it was")
    time.sleep(1)
    print("And they'll just keep on runnin' it")
    time.sleep(1)
    print("forever just because")
    time.sleep(1)
"""


# --------------------------------------
st.header("Getting a Running Start", divider=True)
st.write("If your page only needs one process at a time, use the `spm.run` function, which is very similar to `subprocess.run`.")

with st.echo(code_location="below"):
    import streamlit_process_manager as spm

    # Create a single re-runnable process and loop here until it finishes.
    spm.run(
        [sys.executable, "-u", "-c", counter_prog],
        output_file=Path(".temp/test.output"),
        label=f"A Simple Counter",
        loop=True,
        expanded=True,
        group="use_run",
        rerunable=True,
    )

with st.expander("`counter_prog` used in this example"):
    st.code(counter_prog)

st.header("Staying Single", divider=True)
st.write("If you need a litle more fine-grained control, use the `ProcessManger.single()` pattern.")

with st.echo(code_location="below"):
    import streamlit_process_manager as spm

    # Get the global ProcessManager
    pm = spm.get_manager()

    # Add a single rerunnable process to the ProcessGroup named "single".
    # If this process already exists, from a previous run of your app,
    # a new process won't be created.
    command = [sys.executable, "-c", "-u", counter_prog]
    output_file = Path(".temp/test.output")
    proc = pm.single(spm.RerunableProcess(command, output_file), group="single")

    # Create a ProcessMonitor to view the output and status of your process.
    # This will display a ProcessMonitor widget in your app, but won't
    # actually start following the running process just yet.
    process_label = f"A Simple Counter [pid: {proc.pid}]"
    pmon = spm.st_process_monitor(proc, label=process_label, showcontrols=True)

st.write(
    "Now that you've created a `ProcessMonitor`, you can call `.loop_until_finished()` "
    "to tail the process output while blocking your app. "
)
with st.echo():
    # Continually refresh the process ouptut until it terminates.
    pmon.loop_until_finished()

    # After the process finishes, we check the return code; if it's 0, then we show
    # a happy message in the "contents" section of the process monitor
    if proc.returncode == 0:
        pmon.contents.success("Hooray! The counter finished successfully.")

st.write(
    "\n> Note: If you've just loaded this page, the process probably isn't running yet, "
    "so `.loop_until_finished()` instantly returns. Go ahead and click on the `🟢 Start` button "
    "to start the counter process."
)
with st.expander("`counter_prog` used in this example"):
    st.code(counter_prog)

# ---------------------------------------
st.header("Multi-process", divider=True)

st.write(
    "In more advanced situations, you may want to start multiple processes at a time, and "
    "control and monitor their progress individually. You do this through the global ProcessManager. "
    "object which ensures that your processes continue to be tracked even if the user navigates away "
    "from or refreshes your streamlit page."
)

with st.echo():
    import streamlit_process_manager as spm

    # Get the global ProcessManager
    pm = spm.get_manager()

    # Create a ProcessMonitorGroup for the ProcessGroup called "multi"
    with st.container(border=True):
        st.write("Process Monitors: ")
        pmon_group = spm.st_process_monitor(pm.group("multi"))

st.write(
    """Now set up controls to add and remove processes from the "multi" group.
    If you want, you can also do this programatically from other
    places in your application, they don't need to be exposed via buttons.
    The only key is to make sure that `st_process_monitor` is called
    AFTER you've added all the processes you want."""
)
with st.echo():

    def _add_process():
        new_process_number = len(pm.group("multi")) + 1
        new_process = spm.RerunableProcess(
            cmd=[sys.executable, "-c", endless_prog],
            output_file=Path(f".temp/test{new_process_number}.output"),
            # The above two arguments are the only ones you need. The next two
            # are extra and are here to make this example a bit nicer.
            # We set a custom environment variable PYTHONUNBUFFERED=1 to make "print"
            # statements flush faster.
            env={"PYTHONUNBUFFERED": "1"},
            # And because the endless_prog occasionally clears it's output file,
            # we disable the output_capture_cache
            cache_output_capture=False,
        )
        pm.group("multi").add(new_process)

    cols = st.columns((1, 1, 1))
    cols[0].button("Add Process", on_click=_add_process)
    cols[1].button("Start All Processes", on_click=pm.group("multi").start_all)
    cols[2].button("Remove finished", on_click=pm.group("multi").pop_finished)  # type: ignore[arg-type]

st.write(
    "You're nearly done! As close to the end of the app as possible, we call `loop_until_finished()` to start "
    "monitoring all processes until they all finish."
)

st.code("pmon_group.loop_until_finished()")
with st.expander("`endless_prog` used in this example"):
    st.code(endless_prog)


def _clear_caches():
    pm = spm.get_manager()
    pm.group("single").terminate_all()
    pm.group("multi").terminate_all()
    pm.group("use_run").terminate_all()
    pm.group("single").unsafe_clear()
    pm.group("multi").unsafe_clear()
    pm.group("use_run").unsafe_clear()


st.divider()
st.button("Terminate and Clear all Processes", on_click=_clear_caches, type="primary")

pmon_group.loop_until_finished()
