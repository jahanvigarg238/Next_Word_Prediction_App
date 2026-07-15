# Next-Word Prediction: A From-Scratch Architecture Comparison

Three sequence models — a BiLSTM, a BiGRU with hand-written additive
attention, and a from-scratch decoder-only Transformer — trained on the
same corpus under identical conditions, and compared on perplexity and
top-k accuracy rather than accuracy alone.

## Why this project exists

The standard version of this project (LSTM on Hamlet, single greedy
next-word prediction) is a well-known intro tutorial. This version exists
to demonstrate three things a tutorial version doesn't:

1. **Correct methodology.** The original random train/test split over
   overlapping n-grams leaks context windows between train and test. This
   version splits at the document level *before* generating n-grams, so
   train/val/test come from genuinely disjoint text spans.
2. **Architecture understanding, not just API calls.** The attention
   mechanism and the Transformer block (multi-head self-attention +
   causal masking + positional encoding) are implemented from scratch as
   custom Keras layers, not imported from a pre-built library.
3. **Evaluation that matches how language models are actually judged.**
   Perplexity and top-k accuracy on a held-out split, not training
   accuracy on leaked data.

## Architecture

| Model | Description | Params |
|---|---|---|
| `bilstm` | Embedding → 2×BiLSTM → Dense(softmax) | ~2.1M |
| `bigru_attention` | Embedding → 2×BiGRU → custom additive attention → Dense(softmax) | ~1.9M |
| `mini_transformer` | Embedding + positional encoding → 2× (causal self-attention + FFN) → pooling → Dense(softmax) | ~1.5M |

All three share the same input pipeline (padded token sequences,
`max_sequence_len - 1` timesteps) and the same training regime (Adam,
sparse categorical cross-entropy, early stopping on validation loss).

## Data pipeline

- Source: Shakespeare's *Hamlet* (`data/raw/hamlet.txt`, via
  `nltk.corpus.gutenberg`).
- Cleaning: lowercased, Gutenberg boilerplate stripped, whitespace
  normalized. Stage directions and speaker names are kept — they're
  legitimate structure in the source text, not noise.
- **Leak-free split**: lines are split into contiguous train (80%) /
  val (10%) / test (10%) blocks *first*; n-grams are generated
  independently within each block afterward. No n-gram in val/test
  shares a source line with any n-gram in train.

Run it:
```bash
python src/data_pipeline.py
```

## Training

```bash
pip install -r requirements.txt
python src/train.py --epochs 30 --models bilstm bigru_attention mini_transformer
```

Saves each model + a shared tokenizer to `artifacts/`, and writes
`artifacts/comparison.json` with the results below.

## Results

*(Fill in after running `train.py` to completion — this table is
generated automatically as `artifacts/comparison.json` /
printed to stdout. A short smoke-test run is already verified working;
full training to convergence needs more epochs/time than this
environment's CPU sandbox allows, so run it locally or on Colab.)*

| Model | Params | Train time | Perplexity | Top-1 acc | Top-3 acc | Top-5 acc |
|---|---|---|---|---|---|---|
| bilstm | | | | | | |
| bigru_attention | | | | | | |
| mini_transformer | | | | | | |

## Serving

Single self-contained Streamlit app — no separate backend, which keeps
deployment to Streamlit Community Cloud a one-click "push repo, point at
`streamlit_app.py`" process.

```bash
streamlit run streamlit_app.py
```

Four tabs:
- **Predict** — single-model next-word prediction with a probability bar chart
- **Generate** — multi-word autoregressive generation with temperature/top-k sampling
- **Compare models** — all three architectures' top-5 predictions side by side on the same input
- **Model insights** — PCA projection of the learned embedding space, plus the full metrics table

## Deploying to Streamlit Community Cloud

1. Push this repo to GitHub, **including the trained files in `artifacts/`**
   (`{name}_savedmodel/` folders + `{name}_embedding.npy` + `tokenizer.pickle` + `meta.json` + `comparison.json`) —
   the app expects them to already exist; it doesn't train on startup.
2. On [share.streamlit.io](https://share.streamlit.io), point at the repo,
   branch, and `streamlit_app.py` as the entry point.
3. That's it — no server/Docker to manage.

The `_savedmodel/` folders are a few MB each, well within GitHub's normal file-size
limits, so no LFS needed for this project's scale.

## Tests / CI

```bash
pytest tests/ -v
```
GitHub Actions runs lint + tests on every push (`.github/workflows/ci.yml`).

## Project structure

```
├── data/raw/hamlet.txt
├── src/
│   ├── data_pipeline.py   # cleaning, leak-free split, tokenization
│   ├── models.py          # BiLSTM, BiGRU+attention, mini-Transformer
│   ├── train.py           # training + comparison harness
│   ├── evaluate.py        # perplexity, top-k accuracy
│   └── generate.py        # sampling strategies (greedy/temp/top-k/top-p)
├── streamlit_app.py        # the entire app — predict/generate/compare/insights
├── artifacts/               # trained models + tokenizer (generated by train.py)
├── tests/test_pipeline.py
├── .github/workflows/ci.yml
└── requirements.txt
```

## Why SavedModel, not .h5/.keras, for deployment

Earlier versions of this project loaded trained models directly from
`.h5`/`.keras` files, which store custom layers (the attention layer,
the Transformer block) as references to Python class code. Loading
them back requires *reconstructing* those Python objects — and if the
TensorFlow/Keras version doing the reconstructing differs even
slightly from the version that saved them, that reconstruction step is
exactly where it breaks. This was the repeated "works in Colab, breaks
locally" problem.

The fix: `train.py` additionally exports each model via
`model.export(...)` to TensorFlow's **SavedModel** format — a frozen,
self-contained computation graph. Loading a SavedModel back
(`src/inference.py`, `SavedModelPredictor`) runs the graph directly and
needs **zero** custom layer classes, zero `custom_objects`, and no
import of `src/models.py` at all. This was verified by loading the
exported artifact in an environment with no access to the original
model code and confirming inference still works correctly. This is
the same pattern TensorFlow Serving and most production inference
systems are built on: train with the full Keras API, export once,
deploy the frozen graph everywhere else.

`artifacts/{name}.keras` files are still saved too, as retrainable
checkpoints (useful if you want to fine-tune further) — but the
Streamlit app only ever loads `artifacts/{name}_savedmodel/`.

## Corpus size and prediction quality

The neural models (BiLSTM, BiGRU+attention, mini-Transformer) can
default to predicting the same high-frequency word ("the", "and", "of")
almost regardless of input if trained on too little data or for too
few epochs — the model takes the path of least resistance (always
guess the statistically safest word) before it's learned finer
context-dependent patterns. Hamlet alone (~30K words) is genuinely
small for this; attention-based architectures especially need more
data to move past this.

`data/raw/shakespeare_full.txt` is included as a larger alternative —
a well-known multi-play Shakespeare corpus (~200K words, ~4.5x more
train sequences, ~2.6x larger vocabulary than Hamlet alone). No code
changes needed to use it:

```bash
python src/train.py --data data/raw/shakespeare_full.txt --epochs 30
```

If predictions are still degenerate after training on the larger
corpus for the full epoch budget, check the training/validation loss
curve — if loss is still steadily decreasing at the last epoch, train
longer; if it's plateaued, the model may need more capacity or more
data still.

## Troubleshooting

- **`ModuleNotFoundError`** for any package: run `pip install -r requirements.txt`
  from inside the project folder, and confirm `pip` and `streamlit` point to
  the same Python environment (`where pip` / `where streamlit` on Windows
  should share the same parent folder).
- **Model fails to load / errors mentioning custom layers or Keras version**:
  make sure you're on the version that exports `_savedmodel/` folders
  (run `python src/train.py` from this version of the project) — the
  SavedModel format doesn't require matching TF/Keras versions between
  training and inference the way `.h5`/`.keras` files with custom layers
  did. See "Why SavedModel" below.
- **`Can't reach the API at localhost:8000`**: you're running an old copy of
  `app/frontend.py`. This project is Streamlit-only now — run
  `streamlit run streamlit_app.py`, not anything under `app/`.

## Possible extensions

- Expand the corpus beyond Hamlet (more Shakespeare plays) for a richer
  vocabulary — needs a machine with internet access to pull via
  `nltk.download('gutenberg')`.
- Add a zero-shot perplexity comparison against a pretrained GPT-2 as an
  external reference point.
- t-SNE/UMAP visualization of the learned embedding space.
- Beam search decoding alongside sampling.
