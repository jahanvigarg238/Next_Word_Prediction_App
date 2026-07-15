"""
Wraps a loaded tf.saved_model so it exposes the same .predict(x) interface
as a Keras model. This means generate.py and evaluate.py, which only ever
call model.predict(...), work unchanged regardless of whether the
underlying model is a live Keras object or a version-independent
SavedModel artifact.

Why this matters: a SavedModel is a frozen computation graph. Loading it
back does NOT require reconstructing any custom Python layer classes
(no AdditiveAttention, no TransformerBlock import needed), which is
exactly the step that breaks across TensorFlow/Keras version
differences. This is the same deployment pattern used by TensorFlow
Serving and most production inference systems -- train with the full
Keras API, export once, deploy the frozen graph everywhere else.
"""
import numpy as np
import tensorflow as tf


class SavedModelPredictor:
    def __init__(self, saved_model_dir: str):
        self._loaded = tf.saved_model.load(saved_model_dir)
        self._infer = self._loaded.signatures["serving_default"]
        # The output key varies by model; grab it once.
        self._output_key = list(self._infer.structured_outputs.keys())[0]

    def predict(self, x, verbose: int = 0) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        result = self._infer(tf.constant(x))
        return result[self._output_key].numpy()
