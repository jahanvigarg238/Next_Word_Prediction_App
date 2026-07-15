"""
Train and compare all registered architectures on the leak-free split,
save each trained model + shared tokenizer, and print a comparison table.

Usage:
    python src/train.py --epochs 30 --data data/raw/hamlet.txt
"""

import argparse
import json
import time
import os

import numpy as np
from tensorflow.keras.callbacks import EarlyStopping

from data_pipeline import prepare_dataset, save_tokenizer
from models import MODEL_REGISTRY
from evaluate import compare_models, print_comparison_table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/raw/hamlet.txt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--out_dir", default="artifacts")
    parser.add_argument("--models", nargs="+", default=list(MODEL_REGISTRY.keys()))
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.data, "r") as f:
        raw_text = f.read()

    ds = prepare_dataset(raw_text)
    x_train, y_train = ds["train"]
    x_val, y_val = ds["val"]
    x_test, y_test = ds["test"]
    print(f"total_words={ds['total_words']}  max_sequence_len={ds['max_sequence_len']}")
    print(f"train={x_train.shape}  val={x_val.shape}  test={x_test.shape}")

    save_tokenizer(ds["tokenizer"], os.path.join(args.out_dir, "tokenizer.pickle"))
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump({"total_words": ds["total_words"], "max_sequence_len": ds["max_sequence_len"]}, f)

    trained = {}
    for name in args.models:
        print(f"\n=== Training {name} ===")
        builder = MODEL_REGISTRY[name]
        model = builder(ds["total_words"], ds["max_sequence_len"])
        n_params = model.count_params()

        early_stopping = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
        t0 = time.time()
        model.fit(
            x_train,
            y_train,
            epochs=args.epochs,
            batch_size=args.batch_size,
            validation_data=(x_val, y_val),
            callbacks=[early_stopping],
            verbose=2,
        )
        train_seconds = time.time() - t0

        model.save(os.path.join(args.out_dir, f"{name}.keras"))
        # Export a frozen, version-independent inference artifact.
        # This is what gets deployed -- loading it back needs no custom
        # layer classes at all, unlike the .keras file above.
        model.export(os.path.join(args.out_dir, f"{name}_savedmodel"))
        # Save the embedding weights separately as a plain numpy array.
        # The insights tab visualizes these; saving them independently
        # of the model format means the visualization never depends on
        # being able to reload a live Keras model.
        emb_layer = next(layer for layer in model.layers if "embedding" in layer.name)
        np.save(os.path.join(args.out_dir, f"{name}_embedding.npy"), emb_layer.get_weights()[0])
        trained[name] = (model, n_params, train_seconds)

    print("\n=== Final comparison on held-out test set ===")
    rows = compare_models(trained, x_test, y_test)
    print_comparison_table(rows)
    with open(os.path.join(args.out_dir, "comparison.json"), "w") as f:
        json.dump(rows, f, indent=2)


if __name__ == "__main__":
    main()
