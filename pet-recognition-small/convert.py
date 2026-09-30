"""Convert open-noodle/pet-recognition-small (ONNX) to the LiteRT float16 model Lightforge downloads.

Environment (Python 3.12):
    uv pip install onnx2tf "tensorflow==2.19.*" "tf_keras==2.19.*" onnx onnxruntime \
        onnx_graphsurgeon sng4onnx onnxsim psutil ai_edge_litert numpy pillow

Usage:
    python convert.py model.onnx pet-recognition-small-fp16.tflite [corpus_dir]

`model.onnx` is recognition/model.onnx at commit 9dd4c915be29a81b116b3e30eb996c59d0e7ede0
(sha256 6a5e2373ab348bed588cef4072f3914ca9c8bacde3e8d0651019e8dad86b24ba). When a corpus
directory of `<identity>-<n>.jpg` crops is given, the script compares the result with the
ONNX original and fails if any embedding drifts.
"""

import glob
import os
import subprocess
import sys
import tempfile

import numpy as np
import tensorflow as tf
from PIL import Image

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def convert(onnx_path: str, output_path: str, work: str) -> None:
    fixed = os.path.join(work, "model-fixed.onnx")
    subprocess.run(["onnxsim", onnx_path, fixed, "--overwrite-input-shape", "input:1,3,224,224"], check=True)
    # onnx2tf needs a local calibration sample (its upstream download is gone); values only
    # drive its optional self-check, not the converted weights.
    np.save(os.path.join(work, "calibration_image_sample_data_20x128x128x3_float32.npy"),
            np.random.default_rng(0).random((20, 128, 128, 3), dtype=np.float32))
    # Erf (exact GELU) would otherwise become a Flex op that LiteRT cannot run.
    subprocess.run(["onnx2tf", "-i", fixed, "-o", "saved_model", "-osd", "-rtpo", "Erf"], cwd=work, check=True)

    loaded = tf.saved_model.load(os.path.join(work, "saved_model"))
    serving = loaded.signatures["serving_default"]
    source_input = next(iter(serving.structured_input_signature[1]))

    @tf.function(input_signature=[tf.TensorSpec([1, 224, 224, 3], tf.float32, name="input")])
    def serve(input):  # noqa: A002 - the tensor name is part of the model contract
        return {"embedding": next(iter(serving(**{source_input: input}).values()))}

    converter = tf.lite.TFLiteConverter.from_concrete_functions([serve.get_concrete_function()], loaded)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    with open(output_path, "wb") as out:
        out.write(converter.convert())


def verify(onnx_path: str, tflite_path: str, corpus: str) -> None:
    import onnxruntime as ort
    from ai_edge_litert.interpreter import Interpreter

    files = sorted(glob.glob(os.path.join(corpus, "*.jpg")))
    images = np.stack([
        (np.asarray(Image.open(f).convert("RGB").resize((224, 224), Image.BILINEAR), np.float32) / 255 - MEAN) / STD
        for f in files
    ])
    session = ort.InferenceSession(onnx_path)
    reference = np.concatenate([session.run(None, {"input": x[None].transpose(0, 3, 1, 2)})[0] for x in images])
    runner = Interpreter(model_path=tflite_path).get_signature_runner("serving_default")
    converted = np.stack([runner(input=x[None])["embedding"][0] for x in images])

    labels = np.array([os.path.basename(f).rsplit("-", 1)[0] for f in files])

    def nearest(embeddings):
        similarity = embeddings @ embeddings.T
        np.fill_diagonal(similarity, -9)
        return similarity.argmax(1)

    cosine = (converted * reference).sum(1)
    print(f"{len(files)} images, {len(set(labels))} identities, cosine min {cosine.min():.5f}, "
          f"top-1 {(labels[nearest(converted)] == labels).mean():.3f}, "
          f"same nearest neighbour {(nearest(converted) == nearest(reference)).mean():.3f}")
    if cosine.min() < 0.9999:
        sys.exit("Converted embeddings drifted from the ONNX original")


if __name__ == "__main__":
    onnx_model, tflite_model = sys.argv[1], sys.argv[2]
    with tempfile.TemporaryDirectory() as directory:
        convert(os.path.abspath(onnx_model), os.path.abspath(tflite_model), directory)
    if len(sys.argv) > 3:
        verify(onnx_model, tflite_model, sys.argv[3])
