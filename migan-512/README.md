# MI-GAN 512 (LiteRT float16)

Image inpainting generator used by Lightforge's object eraser. It is a format conversion of
the Places2 512×512 generator from [MI-GAN](https://github.com/Picsart-AI-Research/MI-GAN)
(Picsart AI Research, ICCV 2023).

Lightforge downloads this file only when the user asks for the AI eraser. Photos never
leave the device.

## Release

| | |
| --- | --- |
| Tag | `migan-512-tflite-v1` |
| Asset | `migan-512-fp16.tflite` |
| Size | 14,048,880 bytes |
| SHA-256 | `a27126ba65208314b113ca82fa70d9a9e119a74237037cb4cc63609a3adc4eeb` |

## Source

| | |
| --- | --- |
| Repository | `andraniksargsyan/migan` (Hugging Face) |
| Commit | `406830d0fa60666da0071c342ad2fbc8f30c5c64` |
| File | `migan.onnx` |
| SHA-256 | `593eba0b7e04730f1b61c0a3cbca68d97d8d6a7ff5c6a44a7b9d7fcd880fc5ae` |

## I/O contract

| | |
| --- | --- |
| Signature | `serving_default` |
| Input | `input`, float32 `[1, 512, 512, 4]` (NHWC): `concat(mask - 0.5, (rgb * 2 - 1) * mask)` |
| Output | `output`, float32 `[1, 512, 512, 3]` (NHWC), RGB in `[-1, 1]` |

`rgb` is in `[0, 1]`. `mask` is 1 where a pixel is kept and 0 where the model fills it.
Map the output back with `(y + 1) / 2`. Only the hole should be taken from the output; the
caller composites it over the original.

## Changes from the original

`convert.py` produces the release asset from `migan.onnx`:

1. The published ONNX file is a whole pipeline: uint8 input of any size, with resizing and
   compositing inside. Keep only the 512×512 generator between its normalized input and
   its raw output, then fix the shape (`onnxsim`).
2. Convert with `onnx2tf` to NHWC.
3. Re-export with the TensorFlow Lite converter: float16 weights, float32 input and output,
   and a named `serving_default` signature. The graph uses builtin LiteRT ops only, and
   LiteRT's GPU accelerator runs all of it.

## Parity with the ONNX original

On 17 photos with random brush-stroke masks, compared with the original ONNX pipeline inside
the hole:

| | Minimum PSNR | Largest pixel difference |
| --- | --- | --- |
| Extracted ONNX generator | 51.1 dB | 1 level |
| LiteRT float32 | 51.1 dB | 1 level |
| LiteRT float16 (release asset) | 50.6 dB | 9 levels |

## Performance

Samsung Galaxy S24 Ultra (Snapdragon 8 Gen 3), LiteRT 2.2.0 `benchmark_model`:

| Accelerator | Inference | Peak memory |
| --- | --- | --- |
| GPU | ~40 ms | ~135 MB |
| CPU (XNNPACK, 4 threads) | ~690 ms | ~515 MB |

## License

MIT, like the original model; see [LICENSE](LICENSE) and [ATTRIBUTION.md](ATTRIBUTION.md).
