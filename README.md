# Next-Word Prediction: A From-Scratch Architecture Comparison

Three sequence models — a **BiLSTM**, a **BiGRU with hand-written additive attention**, and a **from-scratch decoder-only mini-Transformer** — trained on the same Shakespeare corpus under identical conditions and compared on **perplexity** and **top-k accuracy**, not accuracy alone.

🔗 **Live demo:** https://nextwordpredictionappbilstm.streamlit.app/

---

## Why this project exists

The standard version of this project (LSTM on Hamlet, single greedy next-word prediction) is a well-known intro tutorial. This version goes further to demonstrate:

1. **Correct methodology** — the naive random train/test split over overlapping n-grams leaks context windows between train and test. This version splits at the *document level* first, so train/val/test come from genuinely disjoint text spans.
2. **Architecture understanding, not just API calls** — the attention mechanism and the Transformer block (multi-head self-attention + causal masking + positional encoding) are implemented from scratch as custom Keras layers.
3. **Evaluation that matches how language models are actually judged** — perplexity and top-k accuracy on a held-out split.
4. **Version-independent deployment** — models are exported to TensorFlow's SavedModel format, avoiding the fragile cross-version loading issues that come with `.h5`/`.keras` files containing custom layers.

## Architecture

| Model | Description | Params |
|---|---|---|
| `bilstm` | Embedding → 2×BiLSTM → Dense(softmax) | ~2.1M |
| `bigru_attention` | Embedding → 2×BiGRU → custom additive attention → Dense(softmax) | ~1.9M |
| `mini_transformer` | Embedding + positional encoding → 2× (causal self-attention + FFN) → pooling → Dense(softmax) | ~1.5M |

All three share the same input pipeline and training regime (Adam, sparse categorical cross-entropy, early stopping on validation loss).

## Data

- **Corpus options**: `data/raw/hamlet.txt` (single play, ~30K words) or `data/raw/shakespeare_full.txt` (multiple plays, ~200K words — recommended, since the neural models need more data to avoid collapsing toward high-frequency-word predictions).
- **Cleaning**: lowercased, boilerplate stripped, whitespace normalized.
- **Leak-free split**: lines are split into contiguous train (80%) / val (10%) / test (10%) blocks *before* n-grams are generated, so no n-gram in val/test shares a source line with any n-gram in train.

## Training

```bash
pip install -r requirements.txt
python src/train.py --data data/raw/shakespeare_full.txt --epochs 30
```

This saves, per model, into `artifacts/`:
- `{name}.keras` — a retrainable checkpoint
- `{name}_savedmodel/` — a frozen, version-independent inference artifact (**this is what the app actually loads**)
- `{name}_embedding.npy` — the learned embedding weights, for visualization
- `tokenizer.pickle`, `meta.json`, `comparison.json` — shared across all models

## Why SavedModel, not `.h5`/`.keras`, for deployment

`.h5`/`.keras` files store custom layers (the attention layer, the Transformer block) as references to Python class code. Loading them back requires reconstructing those Python objects — and if the TensorFlow/Keras version doing the reconstructing differs from the version that saved them, that step is exactly where loading breaks.

The fix: `train.py` exports each model via `model.export(...)` to TensorFlow's **SavedModel** format — a frozen, self-contained computation graph. Loading it back (`src/inference.py`, `SavedModelPredictor`) runs the graph directly and needs **zero** custom layer classes or `custom_objects`. This was verified by loading the exported artifact in an environment with no access to the original model code, confirming inference still works correctly. It's the same pattern TensorFlow Serving and most production inference systems are built on.

## Running the app

```bash
streamlit run streamlit_app.py
```

Four tabs:
- **Predict** — single-model next-word prediction with a probability bar chart
- **Generate** — multi-word autoregressive generation with temperature/top-k sampling
- **Compare models** — all three architectures' top-5 predictions side by side on the same input
- **Model insights** — PCA projection of the learned embedding space, plus the full metrics comparison table

## Deploying to Streamlit Community Cloud

1. Push this repo to GitHub, **including the trained artifacts** — `artifacts/*_savedmodel/`, `artifacts/*_embedding.npy`, `tokenizer.pickle`, `meta.json`, `comparison.json`. (The retraining checkpoints, `artifacts/*.keras`, aren't needed by the app and can be excluded via `.gitignore` to keep the repo smaller.)
2. On [share.streamlit.io](https://share.streamlit.io), point at the repo and set `streamlit_app.py` as the entry point.
3. No server or Docker to manage — Streamlit Cloud handles it.

## Results

*(Generated automatically as `artifacts/comparison.json` after running `train.py` to completion — fill in your actual numbers here once trained.)*

| Model | Params | Train time | Perplexity | Top-1 acc | Top-3 acc | Top-5 acc |
|---|---|---|---|---|---|---|
| bilstm | | | | | | |
| bigru_attention | | | | | | |
| mini_transformer | | | | | | |

## Project structure

```
├── data/raw/
│   ├── hamlet.txt
│   └── shakespeare_full.txt
├── src/
│   ├── data_pipeline.py   # cleaning, leak-free split, tokenization
│   ├── models.py          # BiLSTM, BiGRU+attention, mini-Transformer
│   ├── train.py           # training + SavedModel export + comparison harness
│   ├── evaluate.py        # perplexity, top-k accuracy
│   ├── generate.py        # sampling strategies (greedy/temp/top-k/top-p)
│   └── inference.py       # version-independent SavedModel loader
├── streamlit_app.py         # the entire app — predict/generate/compare/insights
├── artifacts/                # trained models + tokenizer (generated by train.py)
├── tests/test_pipeline.py
├── .github/workflows/ci.yml
└── requirements.txt
```

## Tests / CI

```bash
pytest tests/ -v
```
GitHub Actions runs lint + tests on every push (`.github/workflows/ci.yml`).

## Troubleshooting

- **`ModuleNotFoundError`**: run `pip install -r requirements.txt` from inside the project folder.
- **Model fails to load / custom layer or Keras version errors**: make sure you're loading from `artifacts/{name}_savedmodel/`, not a `.keras` file — see "Why SavedModel" above.
- **Predictions look degenerate (same common word regardless of input)**: likely too few training epochs or too small a corpus. Train longer, or use `shakespeare_full.txt` instead of `hamlet.txt`.

## Possible extensions

- Add a zero-shot perplexity comparison against a pretrained GPT-2 as an external reference point.
- Beam search decoding alongside sampling.
- t-SNE/UMAP as an alternative to PCA for the embedding visualization.
