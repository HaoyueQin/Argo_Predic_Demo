# DenseTNT 复现 — WSL 环境搭建与训练指南

> English version: [wsl-setup-guide.md](wsl-setup-guide.md)

> **目标:** 在 Windows + WSL2 + NVIDIA GPU 上复现 DenseTNT 模型  
> **原始代码:** 仓库根 `src/`（上游 DenseTNT + 本项目改造）  
> **预计总耗时:** 环境搭建 1-2 小时 + 训练 10-20 小时（单 GPU）

---

## 前置检查

在开始之前，先确认以下条件：

### 1. 确认 NVIDIA 显卡型号

打开 PowerShell 或 CMD，运行：

```powershell
nvidia-smi
```

如果能看到显卡信息和驱动版本 → 继续下一步。  
如果提示 "不是内部或外部命令" → 需要先安装 NVIDIA 驱动。

### 2. 确认 Windows 版本

WSL 2 需要 Windows 10 版本 2004+（内部版本 19041+）或 Windows 11。运行：

```powershell
winver
```

---

## Phase 1：安装 WSL 2 + Ubuntu

### Step 1.1：启用 WSL（一条命令）

以**管理员身份**打开 PowerShell，运行：

```powershell
wsl --install
```

这会自动安装 WSL 2 + Ubuntu。安装完成后**重启电脑**。

### Step 1.2：确认 WSL 版本

重启后打开 PowerShell：

```powershell
wsl --version
```

确保显示 WSL 版本为 2.x。如果不是：

```powershell
wsl --set-default-version 2
```

### Step 1.3：首次启动 Ubuntu

重启后，开始菜单会出现 "Ubuntu"。点击启动，第一次会要求创建用户名和密码（记好，后面用 sudo 需要）。

---

## Phase 2：在 WSL Ubuntu 中搭建 Python 环境

### Step 2.1：更新系统 + 安装基础包

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3 python3-pip python3-venv build-essential git
```

### Step 2.2：确认 NVIDIA GPU 在 WSL 中可见

```bash
nvidia-smi
```

如果能看到 GPU 信息 → 完美。如果看不到：
1. 确保 Windows 侧安装了最新的 NVIDIA Game Ready 或 Studio 驱动（WSL 2 不需要在 Linux 里装驱动）
2. 确认用的是 WSL 2 不是 WSL 1

### Step 2.3：安装 CUDA Toolkit（用于 Cython 编译）

WSL 2 里 GPU 驱动是 Windows 侧提供的，但编译 Cython 扩展可能需要 CUDA toolkit 里的编译器：

```bash
# 安装 CUDA 11.8（兼容 PyTorch 1.6+；若使用 requirements_densetnt.txt 的 torch>=2.10，
# 请改装 CUDA 12.x 对应版本）
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-wsl-ubuntu-11-8-local_11.8.0-1_amd64.deb
sudo dpkg -i cuda-repo-wsl-ubuntu-11-8-local_11.8.0-1_amd64.deb
sudo cp /var/cuda-repo-wsl-ubuntu-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt update
sudo apt install -y cuda-toolkit-11-8
```

验证安装：

```bash
nvcc --version
# 应显示 release 11.8
```

### Step 2.4：创建虚拟环境 + 安装 PyTorch

```bash
cd <project_root>
python3 -m venv .venv_densetnt
source .venv_densetnt/bin/activate
```

安装 PyTorch（带 CUDA 支持）：

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

验证 CUDA 可用：

```bash
python3 -c "import torch; print(torch.cuda.is_available())"
# 应输出 True
```

---

## Phase 3：安装 Argoverse API + 地图数据

### Step 3.1：安装依赖

```bash
pip install tqdm matplotlib scipy cython
```

### Step 3.2：安装 Argoverse API

本仓库已自带 Argoverse API（`argoverse-api/` 目录，保留上游完整 git 历史），直接本地安装即可。
注意必须加 `--no-deps`：上游 `setup.py` pin 的旧版本依赖（numpy 1.19 等）会与
`requirements_densetnt.txt` 冲突，跳过依赖解析、只安装包本体（依赖由 requirements 补齐）：

```bash
cd <project_root>
pip install -e argoverse-api --no-deps
```

验证：

```bash
python3 -c "from argoverse.map_representation.map_api import ArgoverseMap; print('OK')"
```

### Step 3.3：下载 Argoverse 1 地图数据

Argoverse 1 地图数据约 2GB，需要从官方下载。ArgoverseMap 加载器通过**仓库相对路径**
`argoverse-api/map_files/` 定位地图（见 `argoverse-api/argoverse/map_representation/map_api.py`），
无需设置任何环境变量：

```bash
cd <project_root>
mkdir -p argoverse-api/map_files
```

下载地图文件（如果官方链接失效，去 https://www.argoverse.org/av1.html 找最新链接）：

```bash
# 下载地图压缩包
wget https://s3.amazonaws.com/argoai-argoverse/hd_maps.tar.gz -O /tmp/hd_maps.tar.gz

# 解压（若压缩包含 map_files/ 顶层目录，直接解压到 argoverse-api/ 下即可）
tar -xzf /tmp/hd_maps.tar.gz -C argoverse-api/

# 若解压结果不在 argoverse-api/map_files/，请把内容移动到该目录
```

解压后的目录结构应该是：

```
argoverse-api/map_files/
  ├── argoverse_HD_maps.json
  ├── MIA_10316.json
  ├── PIT_10315.json
  └── ...
```

---

## Phase 4：准备训练数据 + 编译 Cython

### Step 4.1：组织数据目录

DenseTNT 期望的数据结构：

```
<project_root>/
  └── train/
      └── data/
          ├── 00001.csv
          ├── 00002.csv
          └── ...
  └── val/
      └── data/
          ├── ...
```

将项目中的 `data/raw/` 数据复制过去（80% 训练、20% 验证）：

```bash
cd <project_root>

# 创建目录
mkdir -p train/data val/data

# 把 data/raw 里所有 CSV 复制过来，随机分配到 train/val
python3 << 'EOF'
import os, random, shutil, glob

src = "<project_root>/data/raw"
train_dir = "<project_root>/train/data"
val_dir = "<project_root>/val/data"

files = glob.glob(os.path.join(src, "*.csv"))
random.seed(42)
random.shuffle(files)

split = int(len(files) * 0.8)
train_files = files[:split]
val_files = files[split:]

for f in train_files:
    shutil.copy(f, train_dir)
for f in val_files:
    shutil.copy(f, val_dir)

print(f"Train: {len(train_files)}, Val: {len(val_files)}")
EOF
```

### Step 4.2：编译 Cython 扩展

```bash
cd <project_root>/src
cython -a utils_cython.pyx
python setup.py build_ext --inplace
cd ..
```

---

## Phase 5：启动训练

### Step 5.1：单 GPU 训练命令

```bash
cd <project_root>

OUTPUT_DIR=model_save_full_chunked
GPU_NUM=1

python src/train_v4.py --do_train \
  --data_dir train/data --data_dir_for_val val/data \
  --output_dir model_save_full_chunked \
  --hidden_size 128 --train_batch_size 64 --use_map \
  --core_num 4 --num_workers 0 --use_centerline --distributed_training ${GPU_NUM} \
  --future_frame_num 30 \
  --num_train_epochs 16 --patience 5 \
  --other_params \
    semantic_lane direction l1_loss \
    goals_2D enhance_global_graph subdivide goal_scoring laneGCN point_sub_graph \
    lane_scoring complete_traj complete_traj-3
```

**参数说明：**

| 参数 | 含义 |
|------|------|
| `--distributed_training 1` | 单 GPU 训练（源码默认 8） |
| `--use_map` | 使用地图信息（DenseTNT 核心） |
| `--core_num 4` | 数据加载的 CPU 线程数（笔记本建议 4，太大容易内存不足） |
| `--train_batch_size 64` | 批次大小（单 GPU 时如果显存不够可降到 32 或 16） |

### Step 5.2：训练时间预估

- 官方：8×2080Ti → 每个 epoch 20 分钟 → 16 epochs = 5 小时
- 你的单 GPU（假设 RTX 3060/4060）：每个 epoch 约 40-60 分钟 → 16 epochs = **10-16 小时**
  （注：此为早期小规模实验的估算；60k 数据在 6GB 笔记本 GPU 上的实际训练时长
  以 README 与 `densetnt-overview-zh.md` 中最终实验记录为准）

> 💡 建议：第一次训练先把 epoch 设小一点（`--num_train_epochs 3`），验证能跑通后再跑完整版。

### Step 5.3：如果显存不够 (OOM)

降 batch size，同时调整 `distributed_training` 的设置：

```bash
# 显存 6GB 左右
--train_batch_size 16 --distributed_training 1

# 显存 4GB 左右
--train_batch_size 8 --distributed_training 1
```

---

## Phase 6：评估与导出结果

训练完成后评估：

```bash
cd <project_root>

python src/do_eval.py --argoverse --future_frame_num 30 \
  --do_eval --data_dir val/data --output_dir model_save_full_chunked \
  --hidden_size 128 --use_map --core_num 4 --use_centerline \
  --distributed_training 1 \
  --other_params \
    semantic_lane direction l1_loss \
    goals_2D enhance_global_graph subdivide goal_scoring laneGCN point_sub_graph \
    lane_scoring complete_traj complete_traj-3 \
  --eval_params optimization MRminFDE cnt_sample=9 opti_time=0.1 \
  --model_recover_path model_save_full_chunked/model_save/model.16.bin
```

评估结果会输出 minADE、minFDE、Miss Rate——这些可以和组员跑的 Kalman/LSTM 结果放到同一张对比表里。

---

## 常见问题排查

| 问题 | 原因 | 解决 |
|------|------|------|
| `nvidia-smi` 在 WSL 里无输出 | WSL 1 或驱动太旧 | 更新 Windows NVIDIA 驱动，确认 `wsl --version` 显示 v2 |
| `from argoverse.map_representation...` 报错 | API 未安装或地图数据路径不对 | 确认已 `pip install -e argoverse-api` 且地图在 `argoverse-api/map_files/` |
| Cython 编译报错 | 缺少 C++ 编译器 | `sudo apt install build-essential` |
| 训练 OOM（内存溢出） | batch_size 太大 | 降到 16 或 8 |
| CUDA out of memory | 同上 + 可能模型太大 | 降 batch size |
| 训练速度极慢 | 数据在 Windows 文件系统（/mnt/d/...）上 | 把数据拷贝到 WSL 内部（`/home/user/densetnt_data/`）再训练 |
| ArgoverseMap() 初始化卡住 | 地图文件太大，首次加载慢 | 第一次可能要几分钟加载地图，正常等待 |

> ⚠️ **重要提醒：** 性能瓶颈 — WSL 中通过 `/mnt/d/` 访问 Windows 文件系统 I/O 较慢。如果训练时发现 CPU 利用率很低但速度慢，把 CSV 文件复制到 WSL 的 Linux 原生文件系统（如 `~/densetnt_data/`），速度会显著提升。

---

## 时间规划建议

```
今天（周五晚）  → Phase 1-2：装好 WSL + PyTorch + CUDA
明天（周六）    → Phase 3-4：Argoverse API + 地图数据 + 数据组织 + 编译
周六晚          → Phase 5：启动训练，让它跑一夜
周日            → Phase 6：评估 + 整合结果
```

---

> 💡 **小建议：** 如果 WSL 环境搭到一半卡住了，随时叫我。每一步都有坑，但都能填。
