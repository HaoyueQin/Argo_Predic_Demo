# 项目环境配置说明

> English version: [setup-guide.md](setup-guide.md)

如果你将这个项目复制到了另一台电脑，你需要重新配置 Python 环境。这是因为 Python 虚拟环境和库文件通常包含与特定路径和硬件相关的编译文件，不能直接复制使用。

### 步骤

1. **安装 Python 3.11** — 确保你的新电脑上安装了 Python 3.11（注：环境版本随复现时间变化，以 `requirements*.txt` 与根目录 README 为准）。

2. **创建新虚拟环境** — 在项目根目录下打开终端，运行以下命令：

   ```bash
   python -m venv .venv_argo
   ```

3. **激活虚拟环境**

   - **Windows**:

     ```powershell
     .\.venv_argo\Scripts\Activate.ps1
     ```

   - **Mac/Linux**:

     ```bash
     source .venv_argo/bin/activate
     ```

4. **安装依赖**

   - 使用 `requirements.txt` 文件一键安装**演示环境**（LSTM/Kalman/CV 基线与 Notebook）依赖：

     ```bash
     pip install -r requirements.txt
     ```

   - **DenseTNT 训练环境**请按根目录 README 的说明单独安装：

     ```bash
     pip install -r requirements_densetnt.txt
     pip install -e argoverse-api/ --no-deps
     cd src && cython -a utils_cython.pyx && python setup.py build_ext --inplace
     ```

5. **注册 Jupyter Kernel** — 为了在 Notebook 中使用这个新环境，运行：

   ```bash
   python -m ipykernel install --user --name=argo_env --display-name "Python (Argo Project)"
   ```

### 运行项目

确保你在 Jupyter Notebook 中选择了 `venv_argo` 内核，然后运行 Notebook 或演示脚本即可。
