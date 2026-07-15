"""
Model zoo for the next-word-prediction comparison.

Three architectures, deliberately spanning a spectrum of "how much did
you build vs. call a library layer":

1. build_bilstm        - BiLSTM baseline (canned Keras layers)
2. build_bigru_attention - BiGRU + a hand-written Bahdanau-style additive
                            attention layer (Keras has no built-in
                            "attention over BiGRU outputs" layer, so this
                            one is implemented from scratch as a custom
                            Keras Layer)
3. build_mini_transformer - A small decoder-only Transformer built from
                            scratch: custom positional encoding,
                            multi-head self-attention, and a causal mask,
                            with no reliance on keras_nlp / HF layers.

All three expose the same interface (input: padded int sequence of
length max_sequence_len-1, output: softmax over vocabulary) so they can
be trained and evaluated with the same harness.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential


# ---------------------------------------------------------------------------
# 1. BiLSTM baseline
# ---------------------------------------------------------------------------
def build_bilstm(total_words: int, max_sequence_len: int, embedding_dim: int = 100) -> Model:
    model = Sequential(
        [
            layers.Input(shape=(max_sequence_len - 1,)),
            layers.Embedding(total_words, embedding_dim),
            layers.Bidirectional(layers.LSTM(150, return_sequences=True)),
            layers.Dropout(0.2),
            layers.Bidirectional(layers.LSTM(100)),
            layers.Dense(total_words, activation="softmax"),
        ],
        name="bilstm",
    )
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


# ---------------------------------------------------------------------------
# 2. BiGRU + hand-written additive attention
# ---------------------------------------------------------------------------
@tf.keras.utils.register_keras_serializable(package="next_word_predictor")
class AdditiveAttention(layers.Layer):
    """Bahdanau-style additive attention over a sequence of hidden states,
    pooled into a single context vector. Implemented from scratch (no
    keras.layers.Attention) so the scoring function and weight matrices
    are fully visible / explainable in a writeup.
    """

    def __init__(self, units: int, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.W = layers.Dense(units)
        self.V = layers.Dense(1)

    def call(self, hidden_states):
        # hidden_states: (batch, timesteps, features)
        score = self.V(tf.nn.tanh(self.W(hidden_states)))  # (batch, timesteps, 1)
        weights = tf.nn.softmax(score, axis=1)  # attention over timesteps
        context = tf.reduce_sum(weights * hidden_states, axis=1)  # (batch, features)
        return context, weights

    def get_config(self):
        return {**super().get_config(), "units": self.units}


def build_bigru_attention(total_words: int, max_sequence_len: int, embedding_dim: int = 100) -> Model:
    inputs = layers.Input(shape=(max_sequence_len - 1,))
    x = layers.Embedding(total_words, embedding_dim)(inputs)
    x = layers.Bidirectional(layers.GRU(150, return_sequences=True))(x)
    x = layers.Dropout(0.2)(x)
    hidden_states = layers.Bidirectional(layers.GRU(100, return_sequences=True))(x)
    context, _ = AdditiveAttention(128, name="attention")(hidden_states)
    outputs = layers.Dense(total_words, activation="softmax")(context)
    model = Model(inputs, outputs, name="bigru_attention")
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


# ---------------------------------------------------------------------------
# 3. From-scratch mini decoder-only Transformer
# ---------------------------------------------------------------------------
def positional_encoding(max_len: int, d_model: int) -> tf.Tensor:
    positions = np.arange(max_len)[:, np.newaxis]
    dims = np.arange(d_model)[np.newaxis, :]
    angle_rates = 1 / np.power(10000, (2 * (dims // 2)) / np.float32(d_model))
    angles = positions * angle_rates
    angles[:, 0::2] = np.sin(angles[:, 0::2])
    angles[:, 1::2] = np.cos(angles[:, 1::2])
    return tf.cast(angles[np.newaxis, ...], dtype=tf.float32)


@tf.keras.utils.register_keras_serializable(package="next_word_predictor")
class TransformerBlock(layers.Layer):
    def __init__(self, d_model: int, num_heads: int, ff_dim: int, dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model, self.num_heads, self.ff_dim, self.dropout_rate = d_model, num_heads, ff_dim, dropout
        self.mha = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)
        self.ffn = Sequential(
            [
                layers.Dense(ff_dim, activation="relu"),
                layers.Dense(d_model),
            ]
        )
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.drop1 = layers.Dropout(dropout)
        self.drop2 = layers.Dropout(dropout)

    def call(self, x, training=False):
        seq_len = tf.shape(x)[1]
        causal_mask = tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
        attn_out = self.mha(x, x, attention_mask=causal_mask)
        x = self.norm1(x + self.drop1(attn_out, training=training))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.drop2(ffn_out, training=training))
        return x

    def get_config(self):
        return {
            **super().get_config(),
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "dropout": self.dropout_rate,
        }


def build_mini_transformer(
    total_words: int,
    max_sequence_len: int,
    d_model: int = 128,
    num_heads: int = 4,
    ff_dim: int = 256,
    num_blocks: int = 2,
) -> Model:
    seq_len = max_sequence_len - 1
    inputs = layers.Input(shape=(seq_len,))
    x = layers.Embedding(total_words, d_model)(inputs)
    x = x + positional_encoding(seq_len, d_model)
    for i in range(num_blocks):
        x = TransformerBlock(d_model, num_heads, ff_dim, name=f"transformer_block_{i}")(x)
    # Predict next word from the last non-padded position's representation.
    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(total_words, activation="softmax")(x)
    model = Model(inputs, outputs, name="mini_transformer")
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


MODEL_REGISTRY = {
    "bilstm": build_bilstm,
    "bigru_attention": build_bigru_attention,
    "mini_transformer": build_mini_transformer,
}
