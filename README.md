# X86-GPU_Passthrough-EdgeSync-SDK

**Version:** 2.0 | **Release Date:** May 2025 | **Copyright:** © 2025 Advantech Corporation

## Overview

The **X86-GPU_Passthrough-EdgeSync-SDK** provides a comprehensive environment for Advantech hardware monitoring and GPU Usage. This repository includes two specialized containers:

- **L1-01 Container**: Access Advantech hardware through the EdgeSync-SDK
- **L2-01 Container**: Base Image with NVIDIA GPU passthrough

## Repository Structure

```
X86-GPU_Passthrough-EdgeSync-SDK/
├── L1-01/                # EdgeSync-Adv container
│   ├── build.sh          # Build script for L1-01 container
│   └── docker-compose.yml # Docker configuration
│
└── L2-01/                # Edge-AI-enabled container
    ├── build.sh          # Build script for L2-01 container
    ├── docker-compose.yml # Docker configuration
    ├── wise-test.sh/         # WiseTest 
    └── cuda_diagnostic.sh # CUDA diagnostic script
```

## Installation Requirements


### L1-01 Container Requirements
- Advantech x86 Edge Devices (tested on AIR-520)
- SUSI drivers installed on host system

### L2-01 Container Requirements
- NVIDIA GPU (tested on RTX 6000 Ada Generation)
- NVIDIA driver 550.120 or later
- NVIDIA Container Toolkit 

## Quick Start Guide

### Clone the Repository

```bash
# Clone the repository
git clone https://github.com/Advantech-COE/X86-GPU_Passthrough-EdgeSync-SDK.git
cd X86-GPU_Passthrough-EdgeSync-SDK
```

### Choose Your Container

#### For hardware monitoring (L1-01):

```bash
cd L1-01
chmod +x build.sh
sudo ./build.sh
```

#### For AI development with GPU passthrough (L2-01):

```bash
cd L2-01
chmod +x build.sh
./build.sh
```

## L1-01: EdgeSync-Adv Container

The L1-01 container provides access to Advantech hardware features through the SUSI  APIs.

As in [GitHub - EdgeSync-Adv/advantechiot](https://github.com/EdgeSync-Adv/advantechiot):
### Key Features
- Python interface for hardware sensors and GPIO
- Real-time monitoring of system parameters
- Support for various Advantech industrial computers

### Hardware Support
Tested on:
- Advantech AIR-520 (Ubuntu 22.04)

### Verifying Installation

Once inside the container, verify the installation with:

```bash
# Test hardware connection
#git clone https://github.com/EdgeSync-Adv/advantechiot.git 
# Go inside tests directory and run command as below.
cd /volume/advantech/advantechiot/tests 
python3 -m unittest -v test_advantechiot
```

### Using the advantechiot Package

```python
import advantechiot

# Create a device instance
device = advantechiot.Device()

# Access platform information
print(f"Board name: {device.motherboard.name}")
print(f"BIOS revision: {device.motherboard.bios_revision}")

# Read temperatures
for source in device.motherboard.temperature_sources:
    print(f"{source}: {device.motherboard.get_temperature(source)} °C")

# Control GPIO pins
for gpio_name in device.gpio.pins:
    print(f"{gpio_name} level: {device.gpio.get_level(gpio_name)}")
```

## L2-01: Edge-AI-enabled Container

The Advantech L2-01 container provides a pre-configured CUDA development environment with GPU passthrough capabilities, designed for industrial AI and machine learning applications. This solution simplifies the deployment of GPU-accelerated workloads in edge computing environments while ensuring consistent, reproducible results across development and production environments.

## Validated Hardware Platform

The Advantech L2-01 has been thoroughly tested and validated on the following hardware platform:

- **CPU**: AMD EPYC 7543P 32-Core Processor (64 logical cores)
- **Memory**: 251GB RAM (246GB available)
- **GPU**: 2x NVIDIA RTX 6000 Ada Generation (48GB VRAM each)
- **Operating System**: Ubuntu 22.04.5 LTS (Jammy Jellyfish)
- **Kernel Version**: 6.8.0-59-generic
- **NVIDIA Driver**: 550.144.03
- **CUDA Version**: 12.2
- **cuDNN Version**: 8.9.7

## Test Results

The diagnostic report confirms full functionality of all GPU components:

```
====== FINAL REPORT ======
✅ ALL CHECKS PASSED: Your CUDA environment appears to be properly configured

┌─────────────────────────────────────────────────────┐
│ CUDA Environment Summary                            │
├───────────────────────┬─────────────────────────────┤
│ Overall Status        │ PASS                        │
├───────────────────────┼─────────────────────────────┤
│ NVIDIA Driver         │ 550.144.03                  │
│ CUDA Version          │ 12.2                        │
│ cuDNN Version         │ 8.9.7.*                     │
│ GPU Count             │ 2                           │
├───────────────────────┼─────────────────────────────┤
│ Driver Status         │ ✓ Passed                    │
│ CUDA Toolkit Status   │ ✓ Passed                    │
│ cuDNN Status          │ ✓ Passed                    │
│ CUDA Test Status      │ ✓ Passed                    │
└───────────────────────┴─────────────────────────────┘
```

### GPU Performance Validation

Detail of detected GPUs showing full hardware access and acceleration capabilities:

```
Device 0: NVIDIA RTX 6000 Ada Generation
  Compute capability: 8.9
  Total global memory: 47.50 GB
  Multiprocessors: 142

Device 1: NVIDIA RTX 6000 Ada Generation
  Compute capability: 8.9
  Total global memory: 47.50 GB
  Multiprocessors: 142
```

### NVIDIA Driver Performance

The container has passed all driver functionality tests with full access to the underlying NVIDIA hardware:

```
Wed May  7 02:32:35 2025       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 550.144.03             Driver Version: 550.144.03     CUDA Version: 12.4     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA RTX 6000 Ada Gene...    Off |   00000000:01:00.0 Off |                  Off |
| 33%   62C    P8             41W /  300W |      12MiB /  49140MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
|   1  NVIDIA RTX 6000 Ada Gene...    Off |   00000000:81:00.0  On |                  Off |
| 34%   62C    P8             34W /  300W |     334MiB /  49140MiB |      4%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
```

## GPU Passthrough Capabilities

The Advantech L2-01 container supports full GPU passthrough for:

- NVIDIA RTX/Quadro series (tested with RTX 6000 Ada Generation)
- NVIDIA Tesla/A-series accelerators 
- NVIDIA GeForce RTX/GTX series
- Multi-GPU configurations with automatic load balancing
- Support for CUDA compute capability 8.x to 9.x devices

The container can handle:
- Up to 8 GPUs simultaneously
- Up to 96GB VRAM per GPU
- Mixed GPU types in the same system
- Dynamic GPU allocation

## CUDA and cuDNN Installation Details

### CUDA Installation Paths

The container is configured with the following CUDA paths:

```
CUDA Installation Path: /usr/local/cuda
CUDA Binary Path: /usr/local/cuda/bin
CUDA Library Path: /usr/local/cuda/targets/x86_64-linux/lib
```

### CUDA Library Access

Essential CUDA libraries are properly linked and accessible:

```
libcudart.so -> /usr/local/cuda/targets/x86_64-linux/lib/libcudart.so
libcublas.so -> /usr/local/cuda/targets/x86_64-linux/lib/libcublas.so
libcufft.so -> /usr/local/cuda/targets/x86_64-linux/lib/libcufft.so
libcurand.so -> /usr/local/cuda/targets/x86_64-linux/lib/libcurand.so
libcusparse.so -> /usr/local/cuda/targets/x86_64-linux/lib/libcusparse.so
libcusolver.so -> /usr/local/cuda/targets/x86_64-linux/lib/libcusolver.so
```

Additional libraries included:
- libnvrtc.so - NVIDIA CUDA Runtime Compilation
- libnvjpeg.so - NVIDIA JPEG processing 
- libnvblas.so - NVIDIA BLAS
- libnvToolsExt.so - NVIDIA Tools Extension

### cuDNN Configuration

The container includes cuDNN 8.9.7 with these components:

```
Header location: /usr/include/cudnn.h
Library locations:
  - libcudnn.so.8 -> /usr/lib/x86_64-linux-gnu/libcudnn.so.8
  - libcudnn_ops_train.so.8 -> /usr/lib/x86_64-linux-gnu/libcudnn_ops_train.so.8
  - libcudnn_ops_infer.so.8 -> /usr/lib/x86_64-linux-gnu/libcudnn_ops_infer.so.8
  - libcudnn_cnn_train.so.8 -> /usr/lib/x86_64-linux-gnu/libcudnn_cnn_train.so.8
  - libcudnn_cnn_infer.so.8 -> /usr/lib/x86_64-linux-gnu/libcudnn_cnn_infer.so.8
  - libcudnn_adv_train.so.8 -> /usr/lib/x86_64-linux-gnu/libcudnn_adv_train.so.8
  - libcudnn_adv_infer.so.8 -> /usr/lib/x86_64-linux-gnu/libcudnn_adv_infer.so.8
```

## Pre-installed Development Tools

The Advantech L2-01 container comes with these essential development tools pre-installed:

- **Build Tools**: gcc/g++, make, cmake, build-essential
- **Version Control**: git
- **Python Environment**: Python 3 with pip
- **Utilities**: wget, curl, vim, unzip
- **System Tools**: ca-certificates, gnupg2, lsb-release, software-properties-common

## Production-Ready Features

The container includes several features designed for production deployments:

- **Automatic Restart**: Container configured with `restart: unless-stopped` policy
- **Persistent Storage**: Volume mapping from host to container for data preservation
- **Resource Management**: Shared memory allocation (8GB) optimized for deep learning
- **Error Handling**: Comprehensive error detection in build and verification scripts
- **Version Control**: Support for specific CUDA and cuDNN version selection
- **Environment Persistence**: PATH and LD_LIBRARY_PATH configured for consistent access

## Application Support

The Advantech L2-01 container is optimized for these AI/ML frameworks and applications:

- **Deep Learning Frameworks**: TensorFlow, PyTorch, JAX, MXNet
- **Computer Vision Libraries**: OpenCV, NVIDIA TensorRT
- **HPC Applications**: CUDA-accelerated scientific computing
- **Data Science**: NumPy, SciPy, scikit-learn with GPU acceleration
- **NLP Processing**: BERT, GPT, and transformer models
- **Industrial Applications**: Edge AI, Computer Vision, Predictive Maintenance

## Features

- **Automatic CUDA Detection**: Automatically detects installed CUDA version from your system
- **Flexible Configuration**: Supports command-line configuration of CUDA and cuDNN versions
- **Multi-Version Support**: Compatible with CUDA versions from 11.8 to 12.4 and corresponding cuDNN libraries
- **Complete GPU Passthrough**: Utilizes all available NVIDIA GPUs with full compute capabilities
- **Optimized Memory Management**: Configured with 8GB shared memory for high-performance GPU operations
- **Pre-installed Development Tools**: Ubuntu 22.04 LTS base with essential development packages

## System Requirements

- Host system with NVIDIA GPU(s)
- NVIDIA drivers compatible with CUDA 11.8+
- Docker and Docker Compose
- Linux-based operating system (Ubuntu 20.04+ recommended)

## Quick Start

1. Clone the repository to your local machine
2. Run the build script to detect and configure your CUDA environment:

```bash
./build.sh
```

## Advanced Configuration

### Custom CUDA Version

To specify a custom CUDA version, pass it as an argument to the build script:

```bash
./build.sh 11.8
```

### Custom cuDNN Version

To specify both CUDA and cuDNN versions:

```bash
./build.sh 11.8 8.9.5
```

### Port Configuration

Uncomment the ports section in docker-compose.yml to expose specific ports (e.g., for Jupyter notebooks):

```yaml
ports:
  - "8888:8888"  # For Jupyter notebooks
```

## Verification and Diagnostics

The package includes diagnostic scripts to verify your CUDA installation:

### Basic CUDA Path Verification

Run the CUDA path diagnostic script to check your installation configuration:

```bash
./cuda-diagnostic.sh
```

This script verifies:
- CUDA binary and library paths
- Environment variable configuration
- nvcc compiler availability
- Library dependencies

### Comprehensive CUDA Test

Run the CUDA test script to verify full GPU functionality:

```bash
./cuda-test.sh
```

This script performs:
- NVIDIA driver validation via nvidia-smi
- CUDA toolkit version verification
- cuDNN installation verification
- Compilation and execution of a CUDA test program
- GPU capability detection and reporting

## CUDA Compatibility Matrix

| CUDA Version | Min Driver Version | Compatible cuDNN |
|--------------|-------------------|--------------------|
| CUDA 11.8    | 470.57.02         | cuDNN 8.2.x–8.9.x  |
| CUDA 12.0    | 525.60.13         | cuDNN 8.7.x, 8.8.x |
| CUDA 12.1    | 530.30.02         | cuDNN 8.9.x        |
| CUDA 12.2    | 535.54.03         | cuDNN 8.9.x, 8.10.x|
| CUDA 12.3    | 545.23.06         | cuDNN 8.9.x, 8.10.x|
| CUDA 12.4    | 550.27.05         | cuDNN 9.0.x        |

## Environment Variables

The container is configured with the following key environment variables:

- `NVIDIA_VISIBLE_DEVICES=all`: Makes all GPUs available to the container
- `NVIDIA_DRIVER_CAPABILITIES=compute,utility`: Enables compute and utility functions
- `PATH=/usr/local/cuda/bin:${PATH}`: Ensures CUDA binaries are accessible
- `LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}`: Ensures CUDA libraries are accessible


### Using the Container

#### X11 Forwarding
The container is configured with X11 forwarding to support GUI applications. The build script automatically sets up environment variables:
- XAUTHORITY
- XDG_RUNTIME_DIR

#### Docker Configuration
The container uses the following Docker settings:
- NVIDIA runtime enabled
- Hardware acceleration configured
- Host network mode
- Volumes mounted for persistent storage

### Best Practices

- Pre-allocate GPU memory where possible
- Batch inference for better throughput
- Monitor GPU usage with `nvidia-smi`
- Balance loads between available GPUs

## Troubleshooting

### L1-01 Container Issues

For issues related to the advantechiot package:
- Create an issue on GitHub: https://github.com/EdgeSync-Adv/advantechi

### L2-01 Container Issues

- **GPU Access Issues**: Verify NVIDIA driver installation with `nvidia-smi`
- **CUDA Errors**: Check compatibility between driver and CUDA versions
- **Container Startup Failures**: Ensure NVIDIA Container Toolkit is installed

## Support and Contact

For other issues with these containers:
- Contact Advantech support

---

**Copyright © 2025 Advantech Corporation. All rights reserved.**
