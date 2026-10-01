# Lightforge models

Immutable model files that Lightforge downloads on demand, published as GitHub releases so
the app repository stays free of large binaries. The app pins each asset's URL, size and
SHA-256, so a published asset must never be replaced; ship a new tag instead. Lightforge
never uploads gallery media or search queries.

| Model | Release tag | License | Details |
| --- | --- | --- | --- |
| TinyCLIP semantic search (Balanced, Quality) | `semantic-models-v1` | MIT (TinyCLIP, CLIP) | Signed packages; each archive's `NOTICE.txt` |
| Pet Recognition SMALL, LiteRT float16 | `pet-recognition-small-tflite-v1` | Apache-2.0 | [pet-recognition-small](pet-recognition-small/README.md) |
| MI-GAN 512 inpainting, LiteRT float16 | `migan-512-tflite-v1` | MIT | [migan-512](migan-512/README.md) |

Each model directory holds its license, attributions, and the script that produces the
release asset from the upstream original.
