"""Convert MI-GAN 512 (Places2) from its ONNX pipeline to LiteRT generator models.

Environment (Python 3.12):
    uv pip install onnx2tf "tensorflow==2.19.*" "tf_keras==2.19.*" onnx onnxruntime \
        onnx_graphsurgeon sng4onnx onnxsim psutil ai_edge_litert numpy pillow

Usage:
    python convert.py migan.onnx out_dir

`migan.onnx` is andraniksargsyan/migan (sha256 593eba0b...c5ae, see README). The published
file is a whole pipeline (uint8 in/out, arbitrary size, resize + composite inside). Only the
512x512 generator is kept; the app does resize, normalization and compositing itself.

Generator contract (NHWC after conversion):
    input  "input"  float32 [1, 512, 512, 4] = concat(mask - 0.5, (rgb * 2 - 1) * mask)
                    rgb in [0, 1]; mask 1 = keep, 0 = fill
    output "output" float32 [1, 512, 512, 3] in [-1, 1]; rgb = clip((y + 1) / 2)
"""

import os
import subprocess
import sys
import tempfile

import numpy as np
import onnx
import tensorflow as tf

GENERATOR_IN, GENERATOR_OUT = "222", "1299"  # tensor names inside migan.onnx


def extract(onnx_path: str, work: str) -> str:
    model = onnx.load(onnx_path)
    # The pipeline's dynamic shapes leave the generator boundary without value_info; declare it.
    model.graph.value_info.extend([
        onnx.helper.make_tensor_value_info(GENERATOR_IN, onnx.TensorProto.FLOAT, [1, 4, 512, 512]),
        onnx.helper.make_tensor_value_info(GENERATOR_OUT, onnx.TensorProto.FLOAT, [1, 3, 512, 512]),
    ])
    sub = onnx.utils.Extractor(model).extract_model([GENERATOR_IN], [GENERATOR_OUT])
    sub.graph.input[0].name = "input"
    sub.graph.output[0].name = "output"
    for node in sub.graph.node:
        node.input[:] = ["input" if i == GENERATOR_IN else i for i in node.input]
        node.output[:] = ["output" if o == GENERATOR_OUT else o for o in node.output]
    raw = os.path.join(work, "generator-raw.onnx")
    onnx.save(sub, raw)
    fixed = os.path.join(work, "generator.onnx")
    subprocess.run(["onnxsim", raw, fixed, "--overwrite-input-shape", "input:1,4,512,512"], check=True,
                   stdout=subprocess.DEVNULL)
    return fixed


def convert(fixed: str, out_dir: str, work: str) -> None:
    np.save(os.path.join(work, "calibration_image_sample_data_20x128x128x3_float32.npy"),
            np.random.default_rng(0).random((20, 128, 128, 3), dtype=np.float32))
    subprocess.run(["onnx2tf", "-i", fixed, "-o", "saved_model", "-osd"], cwd=work, check=True,
                   stdout=subprocess.DEVNULL)
    loaded = tf.saved_model.load(os.path.join(work, "saved_model"))
    serving = loaded.signatures["serving_default"]
    source_input = next(iter(serving.structured_input_signature[1]))

    @tf.function(input_signature=[tf.TensorSpec([1, 512, 512, 4], tf.float32, name="input")])
    def serve(input):  # noqa: A002 - the tensor name is part of the model contract
        return {"output": next(iter(serving(**{source_input: input}).values()))}

    for name, fp16 in [("migan-512-fp32.tflite", False), ("migan-512-fp16.tflite", True)]:
        converter = tf.lite.TFLiteConverter.from_concrete_functions([serve.get_concrete_function()], loaded)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        if fp16:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
        with open(os.path.join(out_dir, name), "wb") as out:
            out.write(converter.convert())


if __name__ == "__main__":
    source, out_dir = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
    os.makedirs(out_dir, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out_dir) as directory:
        generator = extract(source, directory)
        convert(generator, out_dir, directory)
        os.replace(generator, os.path.join(out_dir, "migan-512-generator.onnx"))
