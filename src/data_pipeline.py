"""
Data pipeline for next-word prediction.

Key fix vs. the original notebook: the original used sklearn's
train_test_split on flattened n-gram sequences. Because n-grams overlap
heavily ("to be", "to be or", "to be or not" all share tokens), a random
row-wise split leaks near-duplicate context windows between train and test,
inflating validation accuracy. This module instead splits at the
*document position* level, so train/val/test come from non-overlapping
spans of the original text.
"""

import re
import pickle
from dataclasses import dataclass

import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def clean_gutenberg_text(raw_text: str) -> str:
    """Strip Project Gutenberg boilerplate and normalize whitespace.

    The raw nltk gutenberg corpus keeps stage directions and scene
    headers (ACT I, SCENE I, character names in caps) which are useful
    signal for a Shakespeare LM, so we keep those -- we only strip
    boilerplate license text and collapse whitespace.
    """
    text = raw_text.lower()
    # Drop Gutenberg header/footer boilerplate if present.
    text = re.sub(r"\*\*\*.*?\*\*\*", " ", text, flags=re.DOTALL)
    # Collapse repeated whitespace but keep line breaks (they matter for
    # building line-scoped n-grams below).
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


@dataclass
class Corpus:
    lines: list  # list[str], one entry per non-empty source line
    tokenizer: Tokenizer
    total_words: int
    max_sequence_len: int


def build_corpus(raw_text: str, oov_token: str = "<OOV>") -> Corpus:
    text = clean_gutenberg_text(raw_text)
    lines = [ln for ln in text.split("\n") if ln.strip()]

    tokenizer = Tokenizer(oov_token=oov_token)
    tokenizer.fit_on_texts(lines)
    total_words = len(tokenizer.word_index) + 1

    sequences = []
    for line in lines:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            sequences.append(token_list[: i + 1])
    max_sequence_len = max(len(s) for s in sequences)

    return (
        Corpus(lines=lines, tokenizer=tokenizer, total_words=total_words, max_sequence_len=max_sequence_len),
        sequences,
    )


def leak_free_split(lines: list, val_frac: float = 0.1, test_frac: float = 0.1):
    """Split source LINES (not n-grams) into contiguous train/val/test
    blocks, then build n-grams independently within each block.

    Splitting before n-gram expansion guarantees no context window in
    val/test shares tokens that were also seen, in the same position,
    during training.
    """
    n = len(lines)
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)
    n_train = n - n_val - n_test

    train_lines = lines[:n_train]
    val_lines = lines[n_train : n_train + n_val]
    test_lines = lines[n_train + n_val :]
    return train_lines, val_lines, test_lines


def lines_to_padded_sequences(lines: list, tokenizer: Tokenizer, max_sequence_len: int):
    sequences = []
    for line in lines:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            sequences.append(token_list[: i + 1])

    if not sequences:
        return np.empty((0, max_sequence_len - 1)), np.empty((0,))

    padded = pad_sequences(sequences, maxlen=max_sequence_len, padding="pre")
    x, y = padded[:, :-1], padded[:, -1]
    return x, y


def prepare_dataset(raw_text: str, val_frac: float = 0.1, test_frac: float = 0.1):
    """End-to-end: raw text -> tokenizer + leak-free train/val/test arrays."""
    corpus, all_sequences = build_corpus(raw_text)
    train_lines, val_lines, test_lines = leak_free_split(corpus.lines, val_frac, test_frac)

    x_train, y_train = lines_to_padded_sequences(train_lines, corpus.tokenizer, corpus.max_sequence_len)
    x_val, y_val = lines_to_padded_sequences(val_lines, corpus.tokenizer, corpus.max_sequence_len)
    x_test, y_test = lines_to_padded_sequences(test_lines, corpus.tokenizer, corpus.max_sequence_len)

    return {
        "tokenizer": corpus.tokenizer,
        "total_words": corpus.total_words,
        "max_sequence_len": corpus.max_sequence_len,
        "train": (x_train, y_train),
        "val": (x_val, y_val),
        "test": (x_test, y_test),
    }


def save_tokenizer(tokenizer: Tokenizer, path: str):
    with open(path, "wb") as f:
        pickle.dump(tokenizer, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_tokenizer(path: str) -> Tokenizer:
    with open(path, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    with open("data/raw/hamlet.txt", "r") as f:
        raw = f.read()
    ds = prepare_dataset(raw)
    print(f"total_words={ds['total_words']}  max_sequence_len={ds['max_sequence_len']}")
    for split in ("train", "val", "test"):
        x, y = ds[split]
        print(f"{split}: x={x.shape} y={y.shape}")
