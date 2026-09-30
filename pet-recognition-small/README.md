# Pet Recognition SMALL (LiteRT float16)

Individual cat/dog re-identification embeddings used by Lightforge's on-device pet
recognition. It is a format conversion of
[open-noodle/pet-recognition-small](https://huggingface.co/open-noodle/pet-recognition-small),
a frozen [facebook/dinov2-small](https://huggingface.co/facebook/dinov2-small) backbone
with a trained linear projection to 512 dimensions.

Lightforge downloads this file only after the user opts in. Photos never leave the device.

## Release

| | |
| --- | --- |
| Tag | `pet-recognition-small-tflite-v1` |
| Asset | `pet-recognition-small-fp16.tflite` |
| Size | 43,832,152 bytes |
| SHA-256 | `63f88741ce15406e90f6ce2194f4fc5018345f03cb408c97405a714ae1cc5759` |

## Source

| | |
| --- | --- |
| Repository | `open-noodle/pet-recognition-small` |
| Commit | `9dd4c915be29a81b116b3e30eb996c59d0e7ede0` |
| File | `recognition/model.onnx` |
| SHA-256 | `6a5e2373ab348bed588cef4072f3914ca9c8bacde3e8d0651019e8dad86b24ba` |

## I/O contract

| | |
| --- | --- |
| Signature | `serving_default` |
| Input | `input`, float32 `[1, 224, 224, 3]` (NHWC), RGB, ImageNet mean/std normalized |
| Output | `embedding`, float32 `[1, 512]`, L2-normalized |

Compare embeddings with cosine similarity (the dot product, since outputs are unit vectors).

## Changes from the original

Only the format changed. `convert.py` reproduces the release asset byte for byte:

1. Fix the batch size to 1 (`onnxsim`).
2. Convert with `onnx2tf`, replacing `Erf` with its builtin pseudo-operator so no Flex
   (TensorFlow) op is needed.
3. Re-export with the TensorFlow Lite converter: float16 weights, float32 input and output,
   and a named `serving_default` signature. The graph uses builtin LiteRT ops only.

## Parity with the ONNX original

On 64 whole-animal crops of 16 identities:

| | |
| --- | --- |
| Minimum cosine similarity to the ONNX embedding | 0.99997 |
| Top-1 identification | 1.000 (ONNX: 1.000) |
| Same nearest neighbour as ONNX | 1.000 |

## License

Apache-2.0, like the original model; see [LICENSE](LICENSE). Training and evaluation
datasets are listed in [ATTRIBUTION.md](ATTRIBUTION.md).
