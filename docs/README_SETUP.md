# 项目环境配置说明 / Environment Setup Instructions

如果你将这个项目复制到了另一台电脑，你需要重新配置 Python 环境。这是因为 Python 虚拟环境和库文件通常包含与特定路径和硬件相关的编译文件，不能直接复制使用。
If you move this project to another computer, you will need to reconfigure the Python environment. Virtual environments cannot be simply copied because they contain path-specific and hardware-specific binaries.

### 步骤 / Steps:

1.  **安装 Python 3.11 / Install Python 3.11**（注：环境版本随复现时间变化，以 requirements*.txt 与 README 为准）
    *   确保你的新电脑上安装了 Python 3.11。
    *   Ensure Python 3.11 is installed on your new machine.

2.  **创建新虚拟环境 / Create a new virtual environment**
    *   在项目根目录下打开终端，运行以下命令：
    *   Open a terminal in the project root and run:
    ```bash
    python -m venv .venv_argo
    ```

3.  **激活虚拟环境 / Activate the environment**
    *   **Windows**:
        ```powershell
        .\.venv_argo\Scripts\Activate.ps1
        ```
    *   **Mac/Linux**:
        ```bash
        source .venv_argo/bin/activate
        ```

4.  **安装依赖 / Install dependencies**
    *   使用 `requirements.txt` 文件一键安装所有依赖：
    *   Install all dependencies using the generated `requirements.txt`:
    ```bash
    pip install -r requirements.txt
    ```

5.  **注册 Jupyter Kernel / Register Jupyter Kernel**
    *   为了在 Notebook 中使用这个新环境，运行：
    *   To use this environment in Notebooks, run:
    ```bash
    python -m ipykernel install --user --name=argo_env --display-name "Python (Argo Project)"
    ```
   
    ## 完成以上步骤后，就可以运行项目了！

6.  **运行项目 / Run the Project**
    *   确保你在 Jupyter Notebook 中选择了 `venv_argo` 内核。
    *   Run the project in Jupyter Notebook with the `venv_argo` kernel.

