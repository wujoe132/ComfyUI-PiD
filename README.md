# ComfyUI-PiD

Custom ComfyUI nodes for running NVIDIA PiD through ComfyUI's native PixelDiT/PiD model support.

<img width="1058" height="604" alt="ComfyUI-PiD workflow screenshot" src="https://github.com/user-attachments/assets/cc5a9da3-94c6-4546-9574-c8387d5dffdb" />

<img width="4096" height="2048" alt="ComfyUI-PiD example output" src="https://github.com/user-attachments/assets/7ccd55ee-e571-4996-9c9c-4b5cecbb4418" />

This version uses the Comfy-Org repackaged PiD models from:

https://huggingface.co/Comfy-Org/PixelDiT

It no longer uses legacy NVIDIA checkpoint/source loading, Hydra configs, or the old custom model cache.

PiD is not a normal ComfyUI `VAE`. For the supported latent-conditioned PiD path, it needs a latent, a prompt/caption, and a sigma value:

```text
LATENT + caption + sigma -> native ComfyUI PiD -> IMAGE
```

Image-conditioning inputs are not part of the released latent-conditioned PiD decode path. Output size is inferred from the latent grid and selected PiD scale.

## Features

- Direct **PiD Decode** node that returns a ComfyUI `IMAGE`.
- Staged workflow: **PiD Prepare -> PiD Sample -> PiD Finalize**.
- **PiD Sample** runs in-process with ComfyUI-native PiD model loading and the NVIDIA-compatible distilled PiD student sampler.
- **PiD KSampler Capture** for grabbing an intermediate latent and matching sigma.
- **PiD Text Prompt** for sharing one prompt between the generation path and the PiD caption input.
- **PiD Caption Creator** for generating a PiD caption from an input `IMAGE` with a local Qwen captioning model.
- **PiD Empty Latent Image** with backbone-aware latent channel/downscale settings.
- **PiD Upscale** for image-only tiled PiD upscaling with 2x/4x/6x/8x output factors.
- Native ComfyUI model loading through `Comfy-Org/PixelDiT`.
- BF16/FP8 model precision selector with native Comfy-Org files, with BF16 as the recommended quality default.
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
- Enough VRAM for native PiD, especially for `2kto4k` and tiled upscaling.

Current Python package requirements:

```text
huggingface-hub>=0.36.0
transformers>=5.0
accelerate>=1.0
Pillow>=10
```

`requirements.txt` does not install PyTorch, `safetensors`, or ComfyUI runtime packages because ComfyUI usually provides them.

## Required Models

### Core PiD models

The PiD decode/sample nodes download these from `Comfy-Org/PixelDiT` when needed, or you can place them manually. Use `model_precision=bf16` for the BF16 files or `model_precision=fp8` for supported FP8 text encoder plus MXFP8 PiD diffusion files.

Text encoders:

```text
ComfyUI/models/text_encoders/nvidia_pid/gemma_2_2b_it_elm_bf16.safetensors
ComfyUI/models/text_encoders/nvidia_pid/gemma_2_2b_it_elm_fp8_scaled.safetensors
```

Diffusion models used by supported node configurations:

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
```

Existing files in the root `models/diffusion_models` or `models/text_encoders` folders are still accepted. New downloads go into the `nvidia_pid` subfolders.

BF16 is the recommended quality default for PiD output. FP8/MXFP8 is available as a lower-VRAM option for Flux1/Z-Image and for Flux2 `2k`; it can show visible white speckle or high-frequency noise on some systems, so switch `model_precision` back to `bf16` if that happens. SD3, SDXL, and Qwen-Image must use `model_precision=bf16`.

Flux2 and Flux2-Klein `2kto4k` must use the BF16 `_2606` checkpoint. The older Flux2 `2kto4k` MXFP8 file is intentionally rejected because it maps to the pre-`_2606` checkpoint family and can produce color drift/artifacts.

Native PiD uses the NVIDIA distilled 4-step student SDE schedule by default:

```text
[0.999, 0.866, 0.634, 0.342, 0.0]
```

Use `pid_steps=4` for released distilled PiD checkpoints. The native student sampler supports `pid_steps` from `1` to `4`; `1` to `3` are experimental, and values above `4` will fail even if an older workflow exposes them in the UI.

For compatibility with the original NVIDIA PiD nodes, captured low-quality latents are passed to PiD directly as `LQ_latent`. The normal PiD Decode and staged PiD Sample path use the original raw latent conditioning.

### Optional Caption Creator model

`PiD Caption Creator` uses a local Qwen captioning model and can auto-download it when `auto_download=true`:

```text
Qwen/Qwen3.5-0.8B
```

Default local folder:

```text
ComfyUI/models/text_encoders/nvidia_pid/qwen35_caption/
```

This node requires `transformers`, `accelerate`, and `Pillow` from `requirements.txt`. It outputs both `text` and `caption`, so it can feed the normal CLIP/text path and the PiD caption path.

### Optional PiD Upscale VAEs

`PiD Upscale` is image-only, so it needs a matching VAE to encode image tiles into the selected PiD backbone's latent format. With `auto_download=true`, the node can download and place these under `ComfyUI/models/vae/nvidia_pid/`:

| Upscale backbone family | Accepted local names | Auto-download source |
| --- | --- | --- |
| Flux1 / Z-Image | `ae.safetensors` | `Comfy-Org/z_image_turbo`, `split_files/vae/ae.safetensors` |
| Flux2 / Flux2-Klein | `flux2_ae.safetensors`, `flux2-vae.safetensors` | `nvidia/PiD`, `checkpoints/flux2_ae.safetensors` |
| SD3 | `sd3_vae.safetensors`, `diffusion_pytorch_model.safetensors` | `nvidia/PiD`, `checkpoints/sd3_vae/vae/diffusion_pytorch_model.safetensors` |

Root-level `ComfyUI/models/vae` files are also accepted for compatibility, but new auto-downloads go into `models/vae/nvidia_pid`.

## Nodes

| Node | Purpose |
| --- | --- |
| **PiD Decode** | One-node native PiD decode from latent to image. |
| **PiD Text Prompt** | One prompt box with `text` for CLIP and `caption` for PiD. |
| **PiD Caption Creator** | Generates an image caption from `IMAGE` and returns matching `text` and `caption` strings. |
| **PiD KSampler Capture** | KSampler-compatible sampler that returns final latent, captured PiD latent, and sigma. |
| **PiD Prepare** | Prepares latent, caption, native model paths, and metadata on CPU. |
| **PiD Sample** | Runs native PiD sampling in-process from prepared CPU latent data. |
| **PiD Finalize** | Converts native PiD pixel output to ComfyUI `IMAGE`. |
| **PiD Empty Latent Image** | Creates backbone-aware PiD-friendly empty latents from base-resolution presets. |
| **PiD Upscale** | Image-only tiled PiD upscaler with optional caption and 2x/4x/6x/8x output factors. |

## Supported Backbones

| Value | Native PiD file family | Latent channels | Latent downscale | Checkpoints |
| --- | --- | ---: | ---: | --- |
| `zimage` | Flux1 PiD | 16 | 8 | `2k`, `2kto4k` |
| `zimage-turbo` | Flux1 PiD | 16 | 8 | `2k`, `2kto4k` |
| `flux` | Flux1 PiD | 16 | 8 | `2k`, `2kto4k` |
| `flux2` | Flux2 PiD | 128 | 16 | `2k`, `2kto4k` |
| `flux2-klein-4b` | Flux2 PiD | 128 | 16 | `2k`, `2kto4k` |
| `flux2-klein-9b` | Flux2 PiD | 128 | 16 | `2k`, `2kto4k` |
| `sd3` | SD3 PiD | 16 | 8 | `2k`, `2kto4k` |
| `sdxl` | SDXL PiD | 4 | 8 | `2kto4k` |
| `qwenimage` | Qwen-Image PiD | 16 | 8 | `2kto4k` |
| `qwenimage-2512` | Qwen-Image PiD | 16 | 8 | `2kto4k` |

`dinov2` and `siglip` are no longer supported because Comfy-Org does not provide native PiD files for those backbones.

`scale=0` uses the native checkpoint scale, currently `4x` for all supported models. SDXL and Qwen-Image only use the `2kto4k` PiD checkpoint in the native model set.

BF16 Flux2 `2kto4k` uses the newer `pid_flux2_1024_to_4096_4step_2606_bf16.safetensors` file. Do not use FP8/MXFP8 for Flux2 `2kto4k`; v15 rejects that combination.

All currently released native PiD checkpoints are 4-step distilled. The recommended default is `pid_steps=4` and the checkpoint's native scale.

## Staged Workflow

Use staged nodes when you want to separate latent preparation, native sampling, and final image conversion:

```text
PiD KSampler Capture pid_latent -> PiD Prepare latent
PiD Text Prompt caption         -> PiD Prepare caption
PiD Prepare                     -> PiD Sample
PiD Sample                      -> PiD Finalize
PiD Finalize image              -> Save Image
```

You can also use `PiD Caption Creator` in image-to-image or upscale workflows:

```text
Load Image -> PiD Caption Creator caption -> PiD Prepare / PiD Upscale caption
```

Recommended Z-Image capture settings:

```text
steps = 50
sampler_name = euler
scheduler = flowmatch_euler_discrete
flowmatch_shift = 3.0
capture_step = 46
```

`flowmatch_euler_discrete` is the PiD node's built-in Diffusers-style FlowMatch Euler Discrete sigma schedule. Z-Image and Z-Image-Turbo use a FlowMatch Euler Discrete schedule with `shift=3.0`. `euler + beta` remains available as a community/experimental ComfyUI setting, but it is not the exact Diffusers-style schedule.

`PiD KSampler Capture` counts `capture_step` like NVIDIA's `save_xt_steps`: the number of completed LDM denoising passes. For example, `capture_step=46` captures the latent at `sigmas[46]`. If `capture_step` is equal to or greater than the effective sampler step count, the node returns the final clean latent with `sigma=0`.

Suggested capture settings by backbone:

| Backbone | LDM steps | Suggested capture | Recommended latent | Suggested KSampler combo |
| --- | ---: | --- | --- | --- |
| `flux`, `sd3` | 28 | 24 | captured latent | `euler` + `flowmatch_euler_discrete` |
| `sdxl` | 30 | 26 | captured latent | `euler` + Comfy SDXL scheduler (`normal`/model default) |
| `flux2` | 50 | 46 | captured latent | `euler` + `flowmatch_euler_discrete` |
| `flux2-klein-4b`, `flux2-klein-9b` | 4 | 2 or 3 | final `x0` | `euler` + `flowmatch_euler_discrete` |
| `qwenimage`, `qwenimage-2512` | 50 | 44 | captured latent | `euler` + `flowmatch_euler_discrete` |
| `zimage` | 50 | 46 | captured latent | `euler` + `flowmatch_euler_discrete`, `flowmatch_shift=3.0` |
| `zimage-turbo` | 9 | 7 optional | final `x0` | `euler` + `flowmatch_euler_discrete`, `flowmatch_shift=3.0` |

Recommended capture settings are workflow-dependent. Existing Z-Image, Flux2, SD3, SDXL, and Qwen capture workflows should continue to use the same capture node and sigma path.

For Qwen-Image workflows, keep the diffusion model `UNETLoader` `weight_dtype` set to `default`. ComfyUI's `fp8_e4m3fn_fast` path can produce speckled Qwen latents before PiD runs; `PiD KSampler Capture` rejects that combination with a clear error.

`PiD Prepare` accepts only the PiD latent and caption as graph inputs; sigma comes from the captured latent when available or from the manual sigma widget.

## Output Size Guide

```text
512x512 base   + 2k     + scale 4 -> 2048x2048
1024x1024 base + 2kto4k + scale 4 -> 4096x4096
```

Valid `2k` base presets:

```text
512x512, 576x432, 432x576, 624x416, 416x624, 672x384, 384x672, 784x336, 336x784
```

Valid `2kto4k` base presets:

```text
1024x1024, 1024x768, 768x1024, 1008x672, 672x1008, 1024x576, 576x1024, 1008x432, 432x1008
```

The base size is the LDM/base image size implied by the latent grid, not the final PiD output. Flux2-family latents use a 16x downscale, so a 1024x1024 base is a 64x64 latent grid with 128 channels.

## PiD Upscale

`PiD Upscale` is an image-only tiled upscaler. It accepts one graph input (`IMAGE`) and returns one graph output (`IMAGE`). `caption` is optional and can be connected from `PiD Caption Creator`, `PiD Text Prompt`, or any string node.

Main widgets:

| Widget | Values / meaning |
| --- | --- |
| `pid_ckpt_type` | `2k` or `2kto4k` |
| `backbone` | `zimage`, `zimage-turbo`, `flux`, `flux2`, `flux2-klein-4b`, `flux2-klein-9b`, or `sd3` |
| `model_precision` | `bf16` or supported `fp8` combinations |
| `upscale_factor` | `2x`, `4x`, `6x`, `8x` |
| `strength` | Numeric PiD detail-regeneration sigma from `0.0` to `1.0`; default is `0.4` |

Tiling behavior:

| `pid_ckpt_type` | Tile size | Tile overlap | Small-image long-edge prepass |
| --- | ---: | ---: | ---: |
| `2k` | 512 | 64 | 512 |
| `2kto4k` | 1024 | 128 | 1024 |

Each tile is encoded through the selected backbone VAE, PiD-upscaled at the native 4x scale, stitched with raised-cosine overlap blending, then Lanczos-resized to the requested output factor. Images below the profile's small-image long edge are first resized near that size and PiD-upscaled once before tiled upscaling.

SDXL and Qwen-Image backbones are intentionally excluded from `PiD Upscale` because the current implementation only maps image VAEs for Flux1/Z-Image, Flux2/Flux2-Klein, and SD3.

Large outputs can require a lot of VRAM. If a run fails, try:

1. Use a smaller base image or lower output factor.
2. Keep cleanup options enabled in decode/sample workflows.
3. Use the staged workflow to free memory between prepare/sample/finalize.
4. Try FP8/MXFP8 where supported.
5. Restart ComfyUI after CUDA allocator crashes.

## PiD Empty Latent Image

`PiD Empty Latent Image` is a preset wrapper for PiD-friendly base resolutions.

It creates zero latents in the selected backbone's latent format:

```text
[batch, latent_channels, height / latent_downscale, width / latent_downscale]
```

The node exposes:

- `pid_ckpt_type`: `2k` or `2kto4k`.
- `resolution`: a dynamic preset list for the selected checkpoint size class.
- `batch_size`.
- `backbone`: used to choose latent channels, latent downscale, and supported checkpoint types.

Backbone selection matters. Flux2-family empty latents are 128-channel latents with 16x downscale, SDXL empty latents are 4-channel latents with 8x downscale, and most other supported backbones use 16 channels with 8x downscale. The node validates that the selected resolution is one of the trained PiD base presets and that the selected backbone supports the selected checkpoint type.

## Offline Setup

For offline use, download the needed files while online.

Core PiD files:

```bash
hf download Comfy-Org/PixelDiT --local-dir ComfyUI/models --include "diffusion_models/pid_*_bf16.safetensors" "diffusion_models/pid_*_mxfp8.safetensors" "text_encoders/gemma_2_2b_it_elm_bf16.safetensors" "text_encoders/gemma_2_2b_it_elm_fp8_scaled.safetensors"
```

Place downloaded diffusion files under:

```text
ComfyUI/models/diffusion_models/nvidia_pid/
```

Place downloaded PixelDiT text encoders under:

```text
ComfyUI/models/text_encoders/nvidia_pid/
```

Then set `auto_download=false`.

Optional Caption Creator model:

```bash
hf download Qwen/Qwen3.5-0.8B --local-dir ComfyUI/models/text_encoders/nvidia_pid/qwen35_caption
```

Optional PiD Upscale VAEs:

```text
Flux1/Z-Image VAE source: Comfy-Org/z_image_turbo / split_files/vae/ae.safetensors
Local target: ComfyUI/models/vae/nvidia_pid/ae.safetensors

Flux2 VAE source: nvidia/PiD / checkpoints/flux2_ae.safetensors
Local target: ComfyUI/models/vae/nvidia_pid/flux2_ae.safetensors

SD3 VAE source: nvidia/PiD / checkpoints/sd3_vae/vae/diffusion_pytorch_model.safetensors
Local target: ComfyUI/models/vae/nvidia_pid/sd3_vae.safetensors
```

## Example Workflows

Complete backbone-specific workflows are included in `example_workflows/`. They include the generation side plus PiD decode side: model loader, text encoder loader, prompt encode, empty latent, `PiD KSampler Capture`, `PiD Prepare`, `PiD Sample`, `PiD Finalize`, and `Save Image`.

Included complete workflows:

```text
pid_flux_complete.json
pid_flux2_complete.json
pid_flux2_klein_4b_complete.json
pid_flux2_klein_9b_complete.json
pid_qwenimage_complete.json
pid_qwenimage_2512_complete.json
pid_sd3_complete.json
pid_sdxl_complete.json
pid_zimage_complete.json
pid_zimage_turbo_complete.json
pid_image_to_image_2k_complete.json
pid_image_to_image_2kto4k_complete.json
pid_upscale_complete.json
```

The backbone workflows output PiD images directly; no VAE baseline image is used.

Clean-image workflows:

- `pid_image_to_image_2k_complete.json`
- `pid_image_to_image_2kto4k_complete.json`

These use:

```text
LoadImage -> ResizeImagesByLongerEdge -> VAEEncode -> PiDPrepare -> PiDSample -> PiDFinalize
```

Upscale workflow:

- `pid_upscale_complete.json`

This uses:

```text
LoadImage -> PiD Caption Creator -> PiD Upscale -> Save Image
```

## Notes

- This is a community wrapper around NVIDIA PiD support for ComfyUI, not an official NVIDIA or ComfyUI project.
- PiD outputs `IMAGE`, not a ComfyUI `VAE`; no separate image-conditioning input is required for the supported latent-conditioned checkpoints.
- `PiD Upscale` is the exception: it accepts an `IMAGE`, encodes tiles with a matching VAE, and then runs the latent-conditioned PiD path internally.
- NVIDIA's PiD weights and Qwen captioning weights may have separate license/usage terms. Check the relevant model cards before commercial use.
- Final latents with `sigma=0.0` can work. For Z-Image-Turbo and Flux2-Klein, final `x0` is the recommended latent; for normal Z-Image / Flux2 / Qwen-Image, captured intermediate latents usually better match the PiD recipe.
- SDXL captured latents are automatically converted from Comfy/k-diffusion's variance-exploding frame to the VP frame expected by PiD.
- Root-level `models/diffusion_models`, `models/text_encoders`, and `models/vae` files are accepted for compatibility, but new downloads are placed in the `nvidia_pid` subfolders.

## License

This project is released under the MIT License.
