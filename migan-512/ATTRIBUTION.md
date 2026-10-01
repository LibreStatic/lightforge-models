# Attribution

## MI-GAN — MIT

Source: https://github.com/Picsart-AI-Research/MI-GAN
Paper: Sargsyan et al., "MI-GAN: A Simple Baseline for Image Inpainting on Mobile Devices", ICCV 2023.
Copyright (c) 2024 Picsart AI Research (PAIR); see [LICENSE](LICENSE).

The Places2 512 generator was taken from the ONNX export at
https://huggingface.co/andraniksargsyan/migan (MIT). Changes: the generator was extracted
from the export's pipeline and converted to LiteRT; the weights were cast to float16.

## Training data — Places365-Standard

MI-GAN was trained by its authors on Places2 (http://places2.csail.mit.edu). No dataset
images are redistributed here.
