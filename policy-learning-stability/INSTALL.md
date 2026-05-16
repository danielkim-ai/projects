# Installation Guide

This guide separates reproducible installation concerns from the main scientific README. The project targets Python 3.9+ and MuJoCo-backed Gymnasium environments.

## Universal Preparation

Upgrade the packaging toolchain before installing the research dependencies:

```bash
python -m pip install --upgrade pip setuptools wheel
```

This step is particularly important because `grpcio`, a transitive dependency used by Ray, can otherwise attempt a fragile source build on machines with older packaging metadata.

## Apple Silicon (M1/M2/M3)

Create and activate a native ARM64 Python environment:

```bash
python3.9 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

PyTorch will automatically expose the Metal backend when available. The code checks `torch.backends.mps.is_available()` and selects `mps` before falling back to CPU, so no CUDA-specific configuration is required on Apple Silicon.

If MuJoCo fails to initialise, ensure that the Gymnasium MuJoCo extras are installed from the same active environment:

```bash
pip install "gymnasium[mujoco]>=0.28.1"
```

## Windows and Linux

Create an isolated environment and install the dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

On Linux, install system OpenGL runtime packages if MuJoCo rendering is required in headless or server environments. On Windows, use a recent Python distribution and ensure that build tools are available if any wheel is unavailable for the target Python version.

## Ray and Colab Notes

Ray can over-allocate memory or CPUs in constrained notebooks. The training script initialises Ray with defensive head-node settings and modest CPU allocation to avoid common Colab start-up failures.

## Why `linear_operator==0.6.1`

`linear_operator` is pinned to `0.6.1` to keep Gaussian Process tensor algebra compatible with the declared `gpytorch>=1.11` range while avoiding unexpected solver API changes. This makes Bayesian temperature adaptation reproducible across local, Colab and CI-style environments.
