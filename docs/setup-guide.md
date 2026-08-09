# Environment Setup Guide

> Chinese version: [setup-guide-zh.md](setup-guide-zh.md)

If you move this project to another computer, you will need to reconfigure the
Python environment. Virtual environments cannot be simply copied because they
contain path-specific and hardware-specific binaries.

### Steps

1. **Install Python 3.11** — make sure Python 3.11 is installed on your new
   machine. (The pinned version may drift over time; the authoritative source
   is `requirements*.txt` and the root README.)

2. **Create a new virtual environment** — open a terminal in the project root
   and run:

   ```bash
   python -m venv .venv_argo
   ```

3. **Activate the environment**

   - **Windows**:

     ```powershell
     .\.venv_argo\Scripts\Activate.ps1
     ```

   - **Mac/Linux**:

     ```bash
     source .venv_argo/bin/activate
     ```

4. **Install dependencies**

   - Install the **demo environment** dependencies (baselines + notebooks)
     with `requirements.txt`:

     ```bash
     pip install -r requirements.txt
     ```

   - For the **DenseTNT training environment**, follow the root README
     instead:

     ```bash
     pip install -r requirements_densetnt.txt
     pip install -e argoverse-api/ --no-deps
     cd src && cython -a utils_cython.pyx && python setup.py build_ext --inplace
     ```

5. **Register the Jupyter kernel** — to use this environment in Notebooks:

   ```bash
   python -m ipykernel install --user --name=argo_env --display-name "Python (Argo Project)"
   ```

### Run the project

Make sure you select the `venv_argo` kernel in Jupyter Notebook, then run the
notebook or the demo scripts.
