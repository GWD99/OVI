# Brandulate OVI v1

AI-powered video + audio generation platform with a Gradio web interface.

**Website:** [brandulate.com](https://brandulate.com)

## Features

- Text-to-Video (T2V) generation with synchronized audio
- Image-to-Video (I2V) generation with synchronized audio
- Video extension (chain multiple segments together)
- Batch processing (generate from multiple images)
- LoRA support (up to 4 simultaneous LoRAs)
- Preset system (save and load generation settings)
- Multi-GPU support
- Memory optimization (block swap, CPU offload, FP8 quantization)
- Tiled VAE decoding for high-resolution output
- Automatic image cropping and resolution matching
- Metadata logging for all generations

## System Requirements

- **GPU:** NVIDIA GPU with 24GB+ VRAM (48GB+ recommended)
  - Supported: RTX 3090, RTX 4090, RTX 5090, A40, A6000, L40S, RTX 6000 PRO
- **CUDA:** 12.8+
- **cuDNN:** 9.4+
- **Python:** 3.10
- **OS:** Windows 10/11, Ubuntu 20.04+
- **Disk:** 100GB+ for models

## Quick Start

### Windows

1. Run `Windows_Install_and_Update.bat`
2. Run `Windows_Start_App.bat`
3. Open the Gradio URL shown in terminal

### RunPod / Massed Compute

See `Runpod_Instructions_READ.txt` or `Massed_Compute_Instructions_READ.txt`.

## Project Structure

```
OVI/
  premium.py          # Main application (Gradio UI + generation logic)
  ovi/                # Core engine module
    ovi_fusion_engine.py  # Fusion engine for video+audio generation
    configs/          # Model and inference configs
    utils/            # Utility modules (LoRA, I/O, model loading)
  ckpts/              # Model checkpoints (downloaded separately)
  outputs/            # Generated videos
  loras/              # LoRA weight files
  presets/            # Saved generation presets
```

## Models

Models are hosted at [huggingface.co/GWD99/ovi-dependencies](https://huggingface.co/GWD99/ovi-dependencies) and downloaded automatically during installation.

## License

Apache 2.0
