import os
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from data_pipeline import clean_gutenberg_text, leak_free_split, prepare_dataset  # noqa: E402
from generate import sample_next_token  # noqa: E402


SAMPLE_TEXT = """ACT I

SCENE I. Elsinore. A platform before the castle.

Barnardo. Who's there?
Francisco. Nay, answer me: stand, and unfold yourself.
Barnardo. Long live the king!
"""


def test_clean_gutenberg_text_lowercases_and_strips():
    cleaned = clean_gutenberg_text(SAMPLE_TEXT)
    assert cleaned == cleaned.lower()
    assert "   " not in cleaned  # no runs of 3+ spaces within a line


def test_leak_free_split_is_contiguous_and_covers_all_lines():
    lines = [f"line {i}" for i in range(100)]
    train, val, test = leak_free_split(lines, val_frac=0.1, test_frac=0.1)
    assert len(train) + len(val) + len(test) == len(lines)
    # contiguous blocks: train comes first, then val, then test
    assert train == lines[: len(train)]
    assert test == lines[-len(test):]


def test_leak_free_split_train_and_test_share_no_lines():
    lines = [f"line {i}" for i in range(100)]
    train, val, test = leak_free_split(lines)
    assert set(train).isdisjoint(set(test))
    assert set(train).isdisjoint(set(val))


def test_prepare_dataset_shapes_are_consistent():
    ds = prepare_dataset(SAMPLE_TEXT, val_frac=0.2, test_frac=0.2)
    x_train, y_train = ds["train"]
    assert x_train.shape[0] == y_train.shape[0]
    assert x_train.shape[1] == ds["max_sequence_len"] - 1


def test_sample_next_token_greedy_matches_argmax():
    probs = np.array([0.1, 0.7, 0.2])
    assert sample_next_token(probs, temperature=0) == 1


def test_sample_next_token_returns_valid_index():
    probs = np.array([0.1, 0.7, 0.2])
    idx = sample_next_token(probs, temperature=1.0, top_k=2)
    assert idx in (0, 1, 2)
