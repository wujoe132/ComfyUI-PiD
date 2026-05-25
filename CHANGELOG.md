# Changelog

## VRAM offload lean build

- Removed the separate automatic settings module from the package.
- Node registration now exposes only **PiD Decode** and **PiD Text Prompt**.
- Updated README to document manual **PiD Decode** settings only.
- Kept the VRAM offload behavior from the previous build: CPU staging before PiD, Comfy model unload before PiD, and PiD unload after decode.

## VRAM offload build

- Before loading PiD, `PiD Decode` now detaches the input latent and baseline image to CPU.
- `PiD Decode` clears Python references to upstream latent, VAE, and baseline tensors before ComfyUI model unload.
- `_free_cuda_memory()` now also runs Python garbage collection before Comfy/CUDA cache cleanup.
- After a successful PiD decode, the PiD model is removed from this node's private cache and moved off CUDA so the next Z-Image generation does not keep PiD in VRAM.
- The same PiD unload cleanup is attempted after PiD inference errors.
- Added an example workflow that sends `VAEDecode IMAGE` into `PiD Decode baseline_image` and leaves `PiD Decode vae` disconnected.
