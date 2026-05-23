# Installation Guide

This project is intentionally lightweight: the smoke-test workflow runs with PyTorch and the Python standard library, while MuJoCo/Gymnasium are recommended for future offline RL environment integration.

## Linux/macOS

### Option A: venv

```bash
cd projects/trustworthy-offline-rl-via-dp
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python main.py --steps 8 --episodes 18 --delete-episode 3
```

### Option B: conda

```bash
cd projects/trustworthy-offline-rl-via-dp
conda create -n trustworthy-offline-rl python=3.11 -y
conda activate trustworthy-offline-rl
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python main.py --steps 8 --episodes 18 --delete-episode 3
```

For Apple Silicon, prefer the current PyTorch macOS wheels and keep MuJoCo at `3.x` or newer. If GPU acceleration is required on Linux, install the PyTorch wheel matching the CUDA runtime from the official PyTorch selector.

## Windows

Use the Python launcher and keep paths ASCII-only where possible.

```powershell
cd projects\trustworthy-offline-rl-via-dp
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python main.py --steps 8 --episodes 18 --delete-episode 3
```

If PowerShell blocks activation, run the interpreter directly:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe main.py --steps 8
```

MuJoCo notes:

- Newer `mujoco` Python wheels bundle the native runtime for most workflows.
- If using a manual MuJoCo install, add its `bin` directory to `PATH`.
- If rendering fails, check that GPU drivers and Visual C++ runtime packages are installed.

Encoding notes:

- Set the terminal to UTF-8 when inspecting Markdown or logs:

```powershell
chcp 65001
$env:PYTHONUTF8 = "1"
```

## Google Colab

Copy this cell into a notebook:

```python
# 1. Mount Google Drive and move to the project directory.
from google.colab import drive
drive.mount("/content/drive")

%cd /content/drive/MyDrive/trustworthy-offline-rl-via-dp

# 2. Install OS packages and Python dependencies from the unified requirements file.
!apt-get update -qq && apt-get install -y -qq xvfb
%pip install -q -r requirements.txt

# 3. Configure encoding, MuJoCo rendering, and virtual display support.
import os
os.environ["PYTHONUTF8"] = "1"
os.environ["MUJOCO_GL"] = "egl"

from pyvirtualdisplay import Display
display = Display(visible=False, size=(1400, 900))
display.start()

# 4. Verify hardware acceleration and run the prototype smoke test.
import torch
print("CUDA Device Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

!python main.py --steps 8 --episodes 18 --delete-episode 3
```

For a fresh clone inside Colab:

```python
%cd /content
!git clone https://github.com/danielkim-ai/projects.git
%cd /content/projects/trustworthy-offline-rl-via-dp
%pip install -q -r requirements.txt
!python main.py --steps 8 --episodes 18 --delete-episode 3
```

## Smoke-Test Contract

A healthy baseline run prints:

```text
step=00 clip=...
step=07 clip=...
rdp_budget={...}
deleted_episode=... sisa_retrain_shards=[...]
mia_report={...}
```

This confirms the prototype path:

```text
Trajectory DP-SGD training -> episode deletion -> LiSSA unlearning -> MIA tracking
```
