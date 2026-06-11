# ComfyUI-PiD

Custom ComfyUI nodes for running NVIDIA PiD through ComfyUI's native PixelDiT/PiD model support.

This version uses the Comfy-Org repackaged PiD models from:

https://huggingface.co/Comfy-Org/PixelDiT

It no longer uses legacy NVIDIA checkpoint/source loading, Hydra configs, or the old custom model cache.

```text
LATENT + caption + sigma -> native ComfyUI PiD -> IMAGE
```

## Features

- Direct **PiD Decode** node that returns a ComfyUI `IMAGE`.
- Staged workflow: **PiD Prepare -> PiD Sample -> PiD Finalize**.
- **PiD Sample** runs in-process with ComfyUI-native PiD model loading and the NVIDIA-compatible distilled PiD student sampler.
- **PiD KSampler Capture** for grabbing an intermediate latent and matching sigma.
- Native ComfyUI model loading through `Comfy-Org/PixelDiT`.
- BF16/FP8 model precision selector with native Comfy-Org files.
- Auto-download into native ComfyUI model folders under `nvidia_pid` subfolders when `auto_download=true`.

## Install

Clone into `ComfyUI/custom_nodes`:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Merserk/ComfyUI-PiD.git
cd ComfyUI-PiD
python -m pip install -r requirements.txt
```

Restart ComfyUI.

Requirements:

- Recent ComfyUI with native PixelDiT/PiD support.
- Python `>=3.10`.
- NVIDIA CUDA GPU recommended.
- Enough VRAM for native PiD, especially for `2kto4k`.

## Required Models

The node downloads these from `Comfy-Org/PixelDiT` when needed, or you can place them manually. Use `model_precision=bf16` for the BF16 files or `model_precision=fp8` for the FP8 text encoder plus MXFP8 PiD diffusion files.

Text encoder:

```text
ComfyUI/models/text_encoders/nvidia_pid/gemma_2_2b_it_elm_bf16.safetensors
ComfyUI/models/text_encoders/nvidia_pid/gemma_2_2b_it_elm_fp8_scaled.safetensors
```

Diffusion models:

```text
ComfyUI/models/diffusion_models/nvidia_pid/pid_flux1_512_to_2048_4step_bf16.safetensors
ComfyUI/models/diffusion_models/nvidia_pid/pid_flux1_1024_to_4096_4step_bf16.safetensors
ComfyUI/models/diffusion_models/nvidia_pid/pid_flux2_512_to_2048_4step_bf16.safetensors
ComfyUI/models/diffusion_models/nvidia_pid/pid_flux2_1024_to_4096_4step_2606_bf16.safetensors
ComfyUI/models/diffusion_models/nvidia_pid/pid_sd3_512_to_2048_4step_bf16.safetensors
ComfyUI/models/diffusion_models/nvidia_pid/pid_sd3_1024_to_4096_4step_bf16.safetensors
ComfyUI/models/diffusion_models/nvidia_pid/pid_sdxl_1024_to_4096_4step_bf16.safetensors
ComfyUI/models/diffusion_models/nvidia_pid/pid_qwenimage_1024_to_4096_4step_bf16.safetensors
ComfyUI/models/diffusion_models/nvidia_pid/pid_flux1_512_to_2048_4step_mxfp8.safetensors
ComfyUI/models/diffusion_models/nvidia_pid/pid_flux1_1024_to_4096_4step_mxfp8.safetensors
ComfyUI/models/diffusion_models/nvidia_pid/pid_flux2_512_to_2048_4step_mxfp8.safetensors
ComfyUI/models/diffusion_models/nvidia_pid/pid_flux2_1024_to_4096_4step_mxfp8.safetensors
```

Existing files in the root `models/diffusion_models` or `models/text_encoders` folders are still accepted. New downloads go into the `nvidia_pid` subfolders.

BF16 is the recommended quality default for PiD output. FP8/MXFP8 is available as a lower-VRAM option for Flux1 and Flux2, but it can show visible white speckle or high-frequency noise on some systems; switch `model_precision` back to `bf16` if that happens. SD3, SDXL, and QwenImage must use `model_precision=bf16`.

Native PiD uses the NVIDIA distilled 4-step student SDE schedule by default (`[0.999, 0.866, 0.634, 0.342, 0.0]`) while still loading the Comfy-Org `.safetensors` models through ComfyUI. This avoids the over-sharp grain that can appear when the distilled PiD checkpoint is driven through a generic Euler/simple sampler.

For compatibility with the original NVIDIA PiD nodes, captured low-quality latents are passed to PiD directly as `LQ_latent`. The normal PiD Decode and staged PiD Sample path use the original raw latent conditioning.

## Nodes

| Node | Purpose |
| --- | --- |
| **PiD Decode** | One-node native PiD decode from latent to image. |
| **PiD Text Prompt** | One prompt box with `text` for CLIP and `caption` for PiD. |
| **PiD KSampler Capture** | KSampler-compatible sampler that returns final latent, captured PiD latent, and sigma. |
| **PiD Prepare** | Prepares latent, caption, native model paths, and metadata on CPU. |
| **PiD Sample** | Runs native PiD sampling in-process from prepared CPU latent data. |
| **PiD Finalize** | Converts native PiD pixel output to ComfyUI `IMAGE`. |

## Supported Backbones

| Value | Native PiD file family | Latent channels | Checkpoints |
| --- | --- | ---: | --- |
| `zimage` | Flux1 PiD | 16 | `2k`, `2kto4k` |
| `zimage-turbo` | Flux1 PiD | 16 | `2k`, `2kto4k` |
| `flux` | Flux1 PiD | 16 | `2k`, `2kto4k` |
| `flux2` | Flux2 PiD | 128 | `2k`, `2kto4k` |
| `flux2-klein-4b` | Flux2 PiD | 128 | `2k`, `2kto4k` |
| `flux2-klein-9b` | Flux2 PiD | 128 | `2k`, `2kto4k` |
| `sd3` | SD3 PiD | 16 | `2k`, `2kto4k` |
| `sdxl` | SDXL PiD | 4 | `2kto4k` |
| `qwenimage` | QwenImage PiD | 16 | `2kto4k` |
| `qwenimage-2512` | QwenImage PiD | 16 | `2kto4k` |

`dinov2` and `siglip` are no longer supported because Comfy-Org does not provide native PiD files for those backbones.

`scale=0` uses the native checkpoint scale, currently `4x` for all supported models.

BF16 Flux2 `2kto4k` uses the newer `pid_flux2_1024_to_4096_4step_2606_bf16.safetensors` file. FP8 Flux2 `2kto4k` uses `pid_flux2_1024_to_4096_4step_mxfp8.safetensors`.

## Staged Workflow

Use staged nodes when you want to separate latent preparation, native sampling, and final image conversion:

```text
PiD KSampler Capture pid_latent -> PiD Prepare latent
PiD Text Prompt caption         -> PiD Prepare caption
PiD Prepare                     -> PiD Sample
PiD Sample                      -> PiD Finalize
PiD Finalize image              -> Save Image
```

Recommended capture settings are still workflow-dependent. Existing Z-Image, Flux2, SD3, SDXL, and Qwen capture workflows should continue to use the same capture node and sigma path.

For Qwen-Image workflows, keep the diffusion model `UNETLoader` `weight_dtype` set to `default`. ComfyUI's `fp8_e4m3fn_fast` path can produce speckled Qwen latents before PiD runs; `PiD KSampler Capture` rejects that combination with a clear error.

## Output Size Guide

```text
512x512 base   + 2k     + scale 4 -> 2048x2048
1024x1024 base + 2kto4k + scale 4 -> 4096x4096
```

Large outputs can require a lot of VRAM. If a run fails, try:

1. Use a smaller base latent.
2. Keep cleanup options enabled.
3. Use the staged workflow to free memory between prepare/sample/finalize.
4. Restart ComfyUI after CUDA allocator crashes.

## Offline Setup

For offline use, download the needed files while online:

```bash
hf download Comfy-Org/PixelDiT --local-dir ComfyUI/models --include "diffusion_models/pid_*_bf16.safetensors" "diffusion_models/pid_*_mxfp8.safetensors" "text_encoders/gemma_2_2b_it_elm_bf16.safetensors" "text_encoders/gemma_2_2b_it_elm_fp8_scaled.safetensors"
```

Place the downloaded diffusion files under `models/diffusion_models/nvidia_pid` and the text encoder under `models/text_encoders/nvidia_pid`, then set `auto_download=false`.
