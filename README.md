# X86-GPU_Passthrough-EdgeSync-SDK

**Version:** 2.0 | **Release Date:** May 2025 | **Copyright:** © 2025 Advantech Corporation

## Overview

The **X86-GPU_Passthrough-EdgeSync-SDK** provides a comprehensive environment for Advantech hardware monitoring and AI application development. This repository includes two specialized containers:

- **L1-01 Container**: Access Advantech hardware through the EdgeSync-SDK
- **L2-01 Container**: Edge AI development environment with NVIDIA GPU acceleration

## Repository Structure

```
X86-GPU_Passthrough-EdgeSync-SDK/
├── L1-01/                # EdgeSync-Adv container
│   ├── build.sh          # Build script for L1-01 container
│   └── docker-compose.yml # Docker configuration
│
└── L2-01/                # Edge-AI-enabled container
    ├── build.sh          # Build script for L2-01 container
    └── docker-compose.yml # Docker configuration
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

#### For AI development with GPU acceleration (L2-01):

```bash
cd L2-01
chmod +x build.sh
./build.sh
```

## L1-01: EdgeSync-Adv Container

The L1-01 container provides access to Advantech hardware features through the SUSI (Secure, Unified, and Smart Interface) APIs.

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

The L2-01 container provides a development environment for AI applications with full GPU acceleration.

### Key Features
- Full NVIDIA GPU hardware acceleration
- Pre-configured AI development environment
- Support for high-performance deep learning

### Hardware Specifications

| Component | Specification |
|-----------|---------------|
| Target Hardware | NVIDIA RTX 6000 Ada Generation |
| GPU | 2x NVIDIA RTX 6000 Ada Generation with 49140 MiB memory |
| Memory | 49140 MiB per GPU |
| CUDA Version | 12.4 |
| Driver Version | 550.120 |

### Software Components

| Component | Version | Description |
|-----------|---------|-------------|
| CUDA | 12.4.131 | GPU computing platform |
| NVIDIA Driver | 550.120 | Graphics driver |

### Verifying Installation

Once inside the container, verify the GPU and CUDA installation with:

```bash
# Check GPU status
nvidia-smi

# Verify CUDA version
nvcc -V
```

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

- **Module Import Errors**: Verify package installation with `pip list | grep advantechiot`
- **Hardware Detection Problems**: Ensure SUSI drivers are properly installed
- **Permission Issues**: Make sure to run the container with `sudo` for hardware access

### L2-01 Container Issues

- **GPU Access Issues**: Verify NVIDIA driver installation with `nvidia-smi`
- **CUDA Errors**: Check compatibility between driver and CUDA versions
- **Container Startup Failures**: Ensure NVIDIA Container Toolkit is installed

## Support and Contact

For issues related to the advantechiot package:
- Create an issue on GitHub: https://github.com/EdgeSync-Adv/advantechiot

For other issues with these containers:
- Contact Advantech support

---

**Copyright © 2025 Advantech Corporation. All rights reserved.**