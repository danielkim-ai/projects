# Installation Guide

This guide records the intended cross-platform environment setup for **Bayesian RL Meets MCMC**. Phase 1 includes SGLD, VAC, hybrid recalibration, and report extraction skeletons; full MuJoCo training will be added in a later implementation phase.

## A. Windows

WSL2 is strongly recommended. Native Windows support for MuJoCo, PyTorch CUDA wheels, and C++ extension builds is possible, but WSL2 gives a more reproducible Linux-like environment.

### 1. Install WSL2 and Ubuntu

```powershell
wsl --install -d Ubuntu-22.04
wsl --set-default-version 2
```

Restart if requested, then open Ubuntu and update the package index:

```bash
sudo apt update
sudo apt upgrade -y
```

### 2. Install system build dependencies

```bash
sudo apt install -y build-essential cmake git curl wget pkg-config \
  libgl1-mesa-dev libglfw3 libglfw3-dev libglew-dev patchelf \
  python3-dev python3-pip python3-venv
```

These packages cover common C++ build requirements for `mujoco`, `gymnasium`, `grpcio`, and scientific Python extensions.

### 3. Install Miniconda

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc
```

Create the environment:

```bash
conda env create -f setup/environment.yaml
conda activate bayes-rl-mcmc
```

### 4. Install CUDA-compatible PyTorch

If the host has an NVIDIA GPU and WSL2 GPU passthrough is enabled, verify visibility:

```bash
nvidia-smi
```

Then install the appropriate PyTorch CUDA wheel. For CUDA 12.1:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

If no GPU is available, use the CPU wheel:

```bash
pip install torch torchvision torchaudio
```

### 5. Install MuJoCo

Modern MuJoCo is distributed through Python wheels:

```bash
pip install mujoco gymnasium[mujoco]
```

For headless rendering inside WSL2, prefer EGL where available:

```bash
export MUJOCO_GL=egl
```

If rendering fails, install Mesa utilities and test OpenGL availability:

```bash
sudo apt install -y mesa-utils
glxinfo | grep "OpenGL"
```

## B. Apple Silicon

The recommended path is a native `osx-arm64` Miniconda or Miniforge installation, avoiding translated x86 environments unless a dependency explicitly requires them.

### 1. Install Miniforge for arm64

Download the `arm64` installer from the Miniforge releases page, then initialise conda:

```bash
bash Miniforge3-MacOSX-arm64.sh
source ~/.zshrc
```

Confirm the platform:

```bash
conda info | grep platform
```

The result should include `osx-arm64`.

### 2. Create the project environment

```bash
conda env create -f setup/environment.yaml
conda activate bayes-rl-mcmc
```

If dependency solving selects incompatible builds, force the native subdirectory:

```bash
CONDA_SUBDIR=osx-arm64 conda env create -f setup/environment.yaml
```

### 3. Install PyTorch with MPS support

Recent stable PyTorch releases include Metal Performance Shaders support on Apple Silicon:

```bash
pip install torch torchvision torchaudio
```

Check MPS availability:

```python
import torch
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(device)
```

### 4. Resolve MuJoCo and grpcio build issues

Install native build tooling:

```bash
xcode-select --install
brew install cmake pkg-config
```

Then install MuJoCo dependencies:

```bash
pip install mujoco gymnasium[mujoco]
```

If `grpcio` fails to build on ARM, upgrade packaging tools first:

```bash
python -m pip install --upgrade pip setuptools wheel
pip install --no-cache-dir grpcio
```

If MuJoCo rendering reports OpenGL or GLFW issues, use a non-rendering test first and defer visual diagnostics until after the base simulation import succeeds.

## C. Google Colab

Enable a GPU runtime through **Runtime -> Change runtime type -> GPU** before executing the setup cell.

```python
from google.colab import drive
drive.mount("/content/drive")

!git clone https://github.com/danielkim-ai/projects.git
%cd projects/projects/bayesian-rl-meets-mcmc

!python -m pip install --upgrade pip setuptools wheel
!pip install -r setup/requirements.txt
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
!pip install mujoco gymnasium[mujoco]

import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
```

For Colab environments without GPU access, the same snippet falls back to CPU execution. Future training scripts should explicitly accept a `--device` argument and default to this detection logic.
