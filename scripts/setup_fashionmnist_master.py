"""ARISE 2026 — set up the MASTER problem with Fashion-MNIST.

Downloads Fashion-MNIST (Zalando), takes a stratified subsample
(20,000 train / 4,000 test images), saves PNGs to:
  /arise/problems/master/train/    (img0001.jpg ...)
  /arise/problems/master/test/     (img20001.jpg ...)
and writes:
  /arise/problems/master/labels.csv  (id,label)   train labels, public
  /arise/problems/master/test.csv    (id)         test image ids, public
  /arise/problems/master/answer.csv  (id,label)   <- SECRET, never distribute
  /arise/problems/master/sample_submission.csv

All images are 28x28 grayscale, 10 classes (0..9).
Also writes a pixel-based logistic regression baseline for sanity checks.

Usage: /arise/venv/bin/python /arise/scripts/setup_fashionmnist_master.py
"""
import gzip
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backend"))
from config import PROBLEMS_DIR  # noqa: E402

BASE = "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/"
FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}
CACHE = Path("/tmp/fashion_mnist")
CACHE.mkdir(exist_ok=True)

def _download(name):
    dest = CACHE / FILES[name]
    if not dest.exists():
        print(f"Downloading {FILES[name]}...")
        urllib.request.urlretrieve(BASE + FILES[name], dest)
    return dest

def _read_idx_images(path):
    with gzip.open(path, "rb") as f:
        magic = int.from_bytes(f.read(4), "big")
        n = int.from_bytes(f.read(4), "big")
        rows = int.from_bytes(f.read(4), "big")
        cols = int.from_bytes(f.read(4), "big")
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape(n, rows, cols)

def _read_idx_labels(path):
    with gzip.open(path, "rb") as f:
        magic = int.from_bytes(f.read(4), "big")
        n = int.from_bytes(f.read(4), "big")
        return np.frombuffer(f.read(), dtype=np.uint8)

# ---- download & parse ----
tr_imgs = _read_idx_images(_download("train_images"))
tr_lbls = _read_idx_labels(_download("train_labels"))
te_imgs = _read_idx_images(_download("test_images"))
te_lbls = _read_idx_labels(_download("test_labels"))
print(f"Raw: train {tr_imgs.shape}, test {te_imgs.shape}")

# ---- stratified subsample: 20k train / 4k test ----
from sklearn.model_selection import train_test_split  # noqa: E402

train_idx, _ = train_test_split(np.arange(len(tr_imgs)), train_size=20000,
                                random_state=42, stratify=tr_lbls)
test_idx, _ = train_test_split(np.arange(len(te_imgs)), train_size=4000,
                               random_state=42, stratify=te_lbls)

train_imgs, train_labels = tr_imgs[train_idx], tr_lbls[train_idx]
test_imgs, test_labels = te_imgs[test_idx], te_lbls[test_idx]
print(f"Subsample: train {train_imgs.shape}, test {test_imgs.shape}")

# ---- write image folders + CSVs ----
d = PROBLEMS_DIR / "master"
(d / "train").mkdir(parents=True, exist_ok=True)
(d / "test").mkdir(parents=True, exist_ok=True)

# Clear old demo images
for sub in ("train", "test"):
    for p in (d / sub).glob("*.jpg"):
        p.unlink()

def _save(imgs, prefix, folder):
    ids = []
    for i, arr in enumerate(imgs, 1):
        img = Image.fromarray(arr, mode="L")
        img_id = f"img{prefix + i:05d}.jpg"
        img.save(folder / img_id)
        ids.append(img_id)
    return ids

train_ids = _save(train_imgs, 0, d / "train")        # img00001.jpg ...
test_ids = _save(test_imgs, 20000, d / "test")       # img20001.jpg ...

pd.DataFrame({"id": train_ids, "label": train_labels}).to_csv(d / "labels.csv", index=False)
pd.DataFrame({"id": test_ids}).to_csv(d / "test.csv", index=False)          # public
pd.DataFrame({"id": test_ids, "label": test_labels}).to_csv(d / "answer.csv", index=False)  # SECRET
pd.DataFrame({"id": test_ids, "prediction": 0}).to_csv(d / "sample_submission.csv", index=False)

# ---- baseline: logistic regression on flattened pixels (subsample for speed) ----
X = train_imgs.reshape(len(train_imgs), -1).astype(np.float32) / 255.0
X_te = test_imgs.reshape(len(test_imgs), -1).astype(np.float32) / 255.0

# Use a 12k subset for the sanity baseline to keep runtime reasonable
base_n = 12000
lr = LogisticRegression(max_iter=500, solver="lbfgs")
lr.fit(X[:base_n], train_labels[:base_n])
pred = lr.predict(X_te)
baseline = accuracy_score(test_labels, pred)

pd.DataFrame({"id": test_ids, "prediction": pred}).to_csv("/tmp/master_submission.csv", index=False)

print(f"Train images: {len(train_ids)} | Test images: {len(test_ids)}")
print(f"Classes: 10 (0..9) | image size: 28x28 grayscale")
print(f"Baseline (pixel LogisticRegression, 12k train) accuracy: {baseline:.4f}")
print(f"Wrote {d / 'train'}/{d / 'test'}/ labels.csv / test.csv / answer.csv / sample_submission.csv")
print("Baseline submission saved to /tmp/master_submission.csv")
