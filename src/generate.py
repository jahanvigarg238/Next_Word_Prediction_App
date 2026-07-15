"""
Autoregressive text generation with multiple decoding strategies.
The original app only ever predicted a single next word via greedy
argmax. This module adds temperature scaling, top-k, and nucleus
(top-p) sampling, and multi-step generation -- the things that make the
demo actually interesting to play with.
"""

import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences


def _sequence_to_input(tokenizer, text: str, max_sequence_len: int):
    token_list = tokenizer.texts_to_sequences([text.lower()])[0]
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len - 1) :]
    return pad_sequences([token_list], maxlen=max_sequence_len - 1, padding="pre")


def predict_topk(model, tokenizer, text: str, max_sequence_len: int, k: int = 5):
    """Return the top-k (word, probability) predictions for the next word."""
    x = _sequence_to_input(tokenizer, text, max_sequence_len)
    probs = model.predict(x, verbose=0)[0]
    top_idx = np.argsort(-probs)[:k]
    index_to_word = {v: k_ for k_, v in tokenizer.word_index.items()}
    return [(index_to_word.get(i, "<unk>"), float(probs[i])) for i in top_idx]


def sample_next_token(probs: np.ndarray, temperature: float = 1.0, top_k: int = 0, top_p: float = 0.0) -> int:
    """Apply temperature scaling, then optional top-k / nucleus filtering,
    then sample. temperature<=0 falls back to greedy argmax.
    """
    if temperature <= 0:
        return int(np.argmax(probs))

    logits = np.log(np.clip(probs, 1e-9, 1.0)) / temperature
    logits -= logits.max()
    scaled = np.exp(logits)
    scaled /= scaled.sum()

    if top_k and top_k > 0:
        top_idx = np.argsort(-scaled)[:top_k]
        mask = np.zeros_like(scaled)
        mask[top_idx] = scaled[top_idx]
        scaled = mask / mask.sum()

    if top_p and 0 < top_p < 1.0:
        sorted_idx = np.argsort(-scaled)
        cum = np.cumsum(scaled[sorted_idx])
        cutoff = np.searchsorted(cum, top_p) + 1
        keep = sorted_idx[:cutoff]
        mask = np.zeros_like(scaled)
        mask[keep] = scaled[keep]
        scaled = mask / mask.sum()

    return int(np.random.choice(len(scaled), p=scaled))


def generate_text(
    model,
    tokenizer,
    seed_text: str,
    max_sequence_len: int,
    n_words: int = 15,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.0,
) -> str:
    index_to_word = {v: k for k, v in tokenizer.word_index.items()}
    text = seed_text
    for _ in range(n_words):
        x = _sequence_to_input(tokenizer, text, max_sequence_len)
        probs = model.predict(x, verbose=0)[0]
        next_idx = sample_next_token(probs, temperature, top_k, top_p)
        next_word = index_to_word.get(next_idx)
        if not next_word:
            break
        text += " " + next_word
    return text
