# Running on a GPU-enabled cloud

Originally, I was running this project on my M1 Macbook, but the training loop was much too slow on MPS.

These will be notes on setting up the project on a cloud GPU. I may come back and clean this up later.

## Background

### Package Management

Python package management seems to be EXTREMELY fragmented.

At a glance:

- `pip`: A package manager for Python packages. It's the most common package manager for Python.
- `venv`: A tool for creating isolated Python environments. It's built into Python 3.3 and later.
- `poetry`: Maintains dependencies for a specific project. Can install packages like pip, but can also manage package-specific environments like venv.
- `pdm`: Like poetry, but with more of a focus on interop with python standards (e.g. pyproject.toml).
- `rye`: Like pdm, but built in Rust. See this [Github discussion](https://github.com/astral-sh/rye/discussions/6) for context on the tool.
- `conda`/`mamba`: Package manager for scientific / data science packages. Can manage Python packages, non-Python
  packages (e.g. system dependencies hpc libraries may depend on), and environments the packages live in.
- `pixi`: Some kind of abstraction over `conda`.
- `uv`: A package manager meant to be a drop-in replacement for `pip`. It's built in Rust, so it's
  faster than `pip`. `rye`, `pixi`, and maybe a few other tools use `uv` under the hood.

I've found the following combinations of the tools in the wild:

- `pip` + `venv`: The most common combination. `pip` installs packages, `venv` manages environments. Not the most ergonomic,
  they give you the basic building blocks you need for a project: a package manager and an environment manager.
- `poetry` / `pdm` / `rye`: These tools bundle the package manager and environment manager into one tool, so you don't need to
  use the relatively low-level `pip` and `venv` tools. They also have features like lockfiles, which can help ensure that
  projects are reproducible.
- `conda/mamba` + `pip`: You use `conda` to manage the environments and install some packages, and `pip` to install other
  packages. In practice, certain packages are only available on `pip` and not `conda`, so you need to use both. This is the
  setup I've seen most often in data science projects, and is the setup I used for the local development environment.
- `docker` + `pip`: You use `docker` to manage the environments and `pip` to install packages.
- `docker` + one of the above: You use `docker` to manage the environments in conjunction with one of the above tooling.

For this setup, I'll be using `docker` + `pixi`
([docs]([https://github.com/prefix-dev/pixi/blob/main/docs/ide_integration/devcontainer.md)). The main reason is so that I can access `conda` packages while benefiting from the
tighter integration with PyPi that `pixi` provides.

#### An observation on Docker

NVIDIA seems to have first class support for Docker with it's NVIDIA Container Toolkit, while you seem to need to jump through
a few (possibly minor) hoops to get ROCm working with Docker.

Note how the `--gpus` flag in the Docker CLI starts with the the following documentation (emphasis mine).

> The `--gpus` flag allows you to access **NVIDIA GPU** resources. First you need to install the `nvidia-container-runtime`.

This is in contrast to ROCm, which seems to require you to install the ROCm runtime and then use the `--device` flag in the
Docker CLI.

I don't think this is enough to make a decision on which GPU to use, but it's an interesting observation that relates to the
NVIDIA vs AMD competition as well as Docker's corporate strategy.

### Devcontainers

I'm using VSCode, which has a feature called "Dev Containers" that allows you to run your development environment in a
container. This is the main reason I'm using Docker for the cloud GPU environment, since I can use the same setup for
local development and cloud GPU development.

It looks like the Pytorch repository itself includes a [devcontainer](https://github.com/pytorch/pytorch/tree/main/.devcontainer) for reproducibility.

## Host machine Setup

### NVIDIA Drivers (Ubuntu / Azure)

I'm using an NC24ads A100 v4 VM running Ubuntu 24.04 on Azure.

Installation steps ([Azure docs](https://learn.microsoft.com/en-us/azure/virtual-machines/linux/n-series-driver-setup), [Ubuntu docs](https://ubuntu.com/server/docs/nvidia-drivers-installation)):

1. Install the `ubuntu-drivers` utility.

   ```bash
   sudo apt update && sudo apt install -y ubuntu-drivers-common
   ```

2. Install the NVIDIA driver.

   ```bash
    sudo ubuntu-drivers install
   ```

   _Note that the Ubuntu documentation
   suggests using the flag `--gpgpu` to install general
   purpose GPU drivers, but the Azure documentation suggests
   using the `install` command without the flag._

### Docker

[Instructions](https://docs.docker.com/engine/install/ubuntu/#install-using-the-repository) - Nothing special here.

### NVIDIA Container Toolkit

The NVIDIA Container Toolkit allows you to access NVIDIA GPUs from Docker containers.

1. [Install NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
2. Restart Docker
3. Run the following command to test that the NVIDIA Container Toolkit is installed correctly:

```bash
sudo docker run --gpus all nvidia/cuda:12.6.1-devel-ubuntu24.04 nvidia-smi
```

## Dev Container

See [docs](https://pixi.sh/latest/ide_integration/devcontainer/)

Something unexpected: The devcontainer will use your local machine's SSH agent, not the remote host's.
