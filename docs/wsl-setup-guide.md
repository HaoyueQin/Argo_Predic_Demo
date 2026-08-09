# DenseTNT Reproduction — WSL Environment Setup & Training Guide

> **Goal:** Reproduce the DenseTNT model on Windows + WSL2 + NVIDIA GPU
> **Source code:** repo root `src/` (upstream DenseTNT + this project's modifications)
> **Estimated total time:** 1–2 h environment setup + 10–20 h training (single GPU)

---

## Pre-checks

Before you start, confirm the following:

### 1. Confirm the NVIDIA GPU model

Open PowerShell or CMD and run:

```powershell
nvidia-smi
```

If you see the GPU info and driver version → continue to the next step.
If it says "not recognized as an internal or external command" → install the
NVIDIA driver first.

### 2. Confirm the Windows version

WSL 2 requires Windows 10 build 2004+ (build 19041+) or Windows 11. Run:

```powershell
winver
```

---

## Phase 1: Install WSL 2 + Ubuntu

### Step 1.1: Enable WSL (one command)

Open PowerShell **as administrator** and run:

```powershell
wsl --install
```

This automatically installs WSL 2 + Ubuntu. **Reboot** when it finishes.

### Step 1.2: Confirm the WSL version

After rebooting, open PowerShell:

```powershell
wsl --version
```

Make sure it shows WSL version 2.x. If not:

```powershell
wsl --set-default-version 2
```

### Step 1.3: First launch of Ubuntu

After the reboot, "Ubuntu" appears in the Start menu. Click it; on first launch
you will be asked to create a username and password (remember them — `sudo`
needs them later).

---

## Phase 2: Set up the Python environment in WSL Ubuntu

### Step 2.1: Update the system + install base packages

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3 python3-pip python3-venv build-essential git
```

### Step 2.2: Confirm the NVIDIA GPU is visible in WSL

```bash
nvidia-smi
```

If you see GPU info → perfect. If not:
1. Make sure the latest NVIDIA Game Ready or Studio driver is installed on the
   Windows side (WSL 2 does not need drivers installed inside Linux)
2. Make sure you are using WSL 2, not WSL 1

### Step 2.3: Install the CUDA Toolkit (for Cython compilation)

In WSL 2 the GPU driver comes from the Windows side, but compiling Cython
extensions may need the compiler from the CUDA toolkit:

```bash
# Install CUDA 11.8 (compatible with PyTorch 1.6+; if you use the
# torch>=2.10 line from requirements_densetnt.txt, install the matching CUDA 12.x instead)
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-wsl-ubuntu-11-8-local_11.8.0-1_amd64.deb
sudo dpkg -i cuda-repo-wsl-ubuntu-11-8-local_11.8.0-1_amd64.deb
sudo cp /var/cuda-repo-wsl-ubuntu-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt update
sudo apt install -y cuda-toolkit-11-8
```

Verify the installation:

```bash
nvcc --version
# should show release 11.8
```

### Step 2.4: Create a virtual environment + install PyTorch

```bash
cd <project_root>
python3 -m venv .venv_densetnt
source .venv_densetnt/bin/activate
```

Install PyTorch (with CUDA support):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

Verify CUDA is available:

```bash
python3 -c "import torch; print(torch.cuda.is_available())"
# should print True
```

---

## Phase 3: Install the Argoverse API + map data

### Step 3.1: Install dependencies

```bash
pip install tqdm matplotlib scipy cython
```

### Step 3.2: Install the Argoverse API

This repository already bundles the Argoverse API (the `argoverse-api/`
directory, keeping the full upstream git history), so install it locally.
`--no-deps` is required: the old versions pinned by the upstream `setup.py`
(e.g. numpy 1.19) would conflict with `requirements_densetnt.txt`. Skip
dependency resolution and install only the package itself (dependencies are
provided by our requirements file):

```bash
cd <project_root>
pip install -e argoverse-api --no-deps
```

Verify:

```bash
python3 -c "from argoverse.map_representation.map_api import ArgoverseMap; print('OK')"
```

### Step 3.3: Download Argoverse 1 map data

The Argoverse 1 map data is about 2 GB and must be downloaded from the official
source. The `ArgoverseMap` loader locates the maps via the **repo-relative path**
`argoverse-api/map_files/` (see
`argoverse-api/argoverse/map_representation/map_api.py`); no environment
variables are needed:

```bash
cd <project_root>
mkdir -p argoverse-api/map_files
```

Download the map files (if the official link is dead, find the latest link at
https://www.argoverse.org/av1.html):

```bash
# Download the map archive
wget https://s3.amazonaws.com/argoai-argoverse/hd_maps.tar.gz -O /tmp/hd_maps.tar.gz

# Extract (if the archive contains a map_files/ top-level dir, extract straight into argoverse-api/)
tar -xzf /tmp/hd_maps.tar.gz -C argoverse-api/

# If the extracted files are not in argoverse-api/map_files/, move them there
```

The expected layout after extraction:

```
argoverse-api/map_files/
  ├── argoverse_HD_maps.json
  ├── MIA_10316.json
  ├── PIT_10315.json
  └── ...
```

---

## Phase 4: Prepare training data + compile Cython

### Step 4.1: Organize the data directory

The data layout DenseTNT expects:

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

Copy the `data/raw/` data from the project over (80% train / 20% val):

```bash
cd <project_root>

# Create the directories
mkdir -p train/data val/data

# Copy all CSVs from data/raw and randomly split into train/val
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

### Step 4.2: Compile the Cython extension

```bash
cd <project_root>/src
cython -a utils_cython.pyx
python setup.py build_ext --inplace
cd ..
```

---

## Phase 5: Start training

### Step 5.1: Single-GPU training command

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

**Parameter notes:**

| Parameter | Meaning |
|-----------|---------|
| `--distributed_training 1` | Single-GPU training (source defaults to 8) |
| `--use_map` | Use map information (DenseTNT core) |
| `--core_num 4` | CPU threads for data loading (4 is suggested for laptops; too many may exhaust memory) |
| `--train_batch_size 64` | Batch size (on a single GPU lower it to 32 or 16 if VRAM runs out) |

### Step 5.2: Training time estimate

- Official: 8×2080Ti → ~20 min/epoch → 16 epochs ≈ 5 h
- Your single GPU (assume RTX 3060/4060): ~40–60 min/epoch → 16 epochs ≈ **10–16 h**
  (This is an estimate from the early small-scale experiments; the actual
  training time of the 60k split on a 6 GB laptop GPU is recorded in the README
  and `densetnt-overview.md`.)

> 💡 Suggestion: for the first run, keep the epochs small
> (`--num_train_epochs 3`) to verify the pipeline, then run the full version.

### Step 5.3: If VRAM runs out (OOM)

Lower the batch size, and adjust `distributed_training` accordingly:

```bash
# ~6 GB VRAM
--train_batch_size 16 --distributed_training 1

# ~4 GB VRAM
--train_batch_size 8 --distributed_training 1
```

---

## Phase 6: Evaluation and exporting results

Evaluate after training:

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

The evaluation outputs minADE, minFDE and Miss Rate — these can be put into the
same comparison table as the Kalman/LSTM results.

---

## Troubleshooting

| Problem | Cause | Solution |
|---------|-------|----------|
| `nvidia-smi` prints nothing in WSL | WSL 1 or outdated driver | Update the Windows NVIDIA driver and confirm `wsl --version` shows v2 |
| `from argoverse.map_representation...` fails | API not installed or wrong map path | Confirm `pip install -e argoverse-api` and maps under `argoverse-api/map_files/` |
| Cython compilation error | Missing C++ compiler | `sudo apt install build-essential` |
| Training OOM (memory) | batch_size too large | Lower it to 16 or 8 |
| CUDA out of memory | Same + possibly the model is too big | Lower batch size |
| Training extremely slow | Data sits on the Windows filesystem (`/mnt/d/...`) | Copy the data inside WSL (e.g. `/home/user/densetnt_data/`) and retrain |
| `ArgoverseMap()` init hangs | Map files are large; first load is slow | The first load can take a few minutes; wait normally |

> ⚠️ **Important:** performance bottleneck — accessing the Windows filesystem
> through `/mnt/d/` in WSL is slow I/O. If training is slow with low CPU
> utilization, copy the CSVs to a native Linux filesystem in WSL (e.g.
> `~/densetnt_data/`) — the speedup is significant.

---

## Suggested time plan

```
Tonight (Fri)    → Phase 1-2: WSL + PyTorch + CUDA
Tomorrow (Sat)   → Phase 3-4: Argoverse API + map data + data organization + compilation
Sat night        → Phase 5: start training, let it run overnight
Sunday           → Phase 6: evaluation + integrate results
```

---

> 💡 **Tip:** if you get stuck halfway through the WSL setup, ask. Every step
> has pitfalls, but all of them can be fixed.
