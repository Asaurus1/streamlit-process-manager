# Streamlit Process Manager

[![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/asaurus1/streamlit-process-manager/pdm.yml)](https://github.com/Asaurus1/streamlit-process-manager/actions)
[![GitHub last commit](https://img.shields.io/github/last-commit/asaurus1/streamlit-process-manager)](https://github.com/Asaurus1/streamlit-process-manager)
[![PyPI - Version](https://img.shields.io/pypi/v/streamlit-process-manager)](https://pypi.org/project/streamlit-process-manager/)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/streamlit-process-manager)
![PyPI - Wheel](https://img.shields.io/pypi/wheel/streamlit-process-manager)
![PyPI - Implementation](https://img.shields.io/pypi/implementation/streamlit-process-manager)
![PyPI - Downloads](https://img.shields.io/pypi/dm/streamlit-process-manager)
[![Licence](https://img.shields.io/badge/licence-Apache%202.0-blue)](https://www.apache.org/licenses/LICENSE-2.0)
[![Streamlit Version](https://img.shields.io/badge/Streamlit->=1.30.0-blue)](https://github.com/streamlit/streamlit)

Let's be honest, trying to manage external processes or subprocesses from a [Streamlit](https://github.com/streamlit/streamlit) app is tricky. The execution model makes it difficult to keep track of which processes are supposed to be running and which ones aren't. Half the time you end up spawning twenty different processes all trying to access the same file and stomping all over one another.

Streamlit Process Manager attempts to make a that a bit easier. It looks something like this:

![Streamlit Process Monitor Animation](https://raw.githubusercontent.com/asaurus1/streamlit-process-manager/main/.github/images/ProcessMonitor.gif)

And here's the simplest program (you have to write `my_program.py` yourself):
```python
import streamlit_process_manager as spm

my_program = ["python", "-u", "my_program.py"]
output_file = "program_output.txt"
process = spm.run(my_program, output_file=output_file)

# in my_program.py
# -------------------------------------------
import time

try:
    for x in range(100):
        print(f"Running... {x}%")
        time.sleep(0.1)
finally:
    print("I'm done!")
```

> ⚠️ **DISCLAIMER** ⚠️
>
> Runing processes on your machine from a public-facing
  application like a Streamlit app can be _incredibly_ risky and comes
  with all sorts of security considerations. Use this project on public apps
  at your own risk, and **never** inject user-provided values or arguments
  into your processes without serious consideration of all the possible injection
  attacks you might be opening your server up to.

## Installation

Alpha: `pip install streamlit-process-manager==0.0.1a`

Latest: `pip install git+https://github.com/Asaurus1/streamlit-process-manager`

Requires
* `Python 3.8.*` or above
* `Streamlit 1.30.*` or above
* `psutil 5.9.8` or above


## Development

This project uses [`pdm`](https://pdm-project.org/) for most of its development tooling. Recommended environment for development is kept in the `pdm.lock` file.

Ensure you have `python3.12` installed, then run
```
pip install pdm
pdm venv create 3.12
pdm install --dev --group test --group lint
```

to get started, then run
```
pdm check
```

to verify that everything is set up and ready to go.

### Testing with tox

If you have multiple versions of Python available on your system you can use `tox` to run tests
on all of them at once. `tox` should be installed when you install the `--dev` dependencies so
you can just run it. Any Python's not available on your system will be skipped.

```
pdm run tox
```

## Licence - Apache 2.0
Copyright 2024 by Alexander Martin

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
