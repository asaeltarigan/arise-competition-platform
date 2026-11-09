"""ARISE 2026 — set up the MASTER problem with APS Failure at Scania Trucks.

Downloads the UCI Scania APS dataset (60,000 train rows, 171 features, binary
class with ~1.7% positive), does a stratified 80/20 split, and writes:
  /arise/problems/master/train.csv   (id, 171 features..., label 0/1)
  /arise/problems/master/test.csv    (id, 171 features...)
  /arise/problems/master/answer.csv  (id, label)  <- SECRET, never distribute
  /arise/problems/master/sample_submission.csv

Metric for this problem is f1_pos (F1 of the positive class) because the data
is heavily imbalanced — accuracy would be meaningless (98% by predicting all 0).

Also writes an XGBoost-style baseline (HistGradientBoosting, class-weighted)
with threshold tuning on a validation slice.

Usage: /arise/venv/bin/python /arise/scripts/setup_scania_master.py
"""
import sys
import zipfile
import urllib.request
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backend"))
from config import PROBLEMS_DIR  # noqa: E402

URL = "https://archive.ics.uci.edu/static/public/421/aps+failure+at+scania+trucks.zip"
LOCAL = Path("/tmp/scania.zip")

if not LOCAL.exists():
    print("Downloading Scania APS dataset (~45MB)...")
    urllib.request.urlretrieve(URL, LOCAL)

with zipfile.ZipFile(LOCAL) as z:
    names = z.namelist()
    print("zip entries:", names)
    train_member = next(n for n in names if "training_set" in n)

    raw = z.read(train_member).decode("latin-1")
    lines = raw.splitlines()
    hdr = next(i for i, l in enumerate(lines) if l.startswith("class,"))
    df = pd.read_csv(__import__("io").StringIO("\n".join(lines[hdr:])),
                     na_values=["na"])
    print("raw shape:", df.shape)

# 171 features, class is 'neg'/'pos' -> map to 0/1
FEATURES = [c for c in df.columns if c != "class"]
df = df.rename(columns={"class": "label"})
df["label"] = df["label"].map({"neg": 0, "pos": 1})
print("class distribution:", df["label"].value_counts().to_dict())

# numeric coercion (all features are numeric; any stray strings to NaN)
for c in FEATURES:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df.insert(0, "id", range(1, len(df) + 1))
df = df[["id"] + FEATURES + ["label"]]

train, test = train_test_split(df, test_size=0.2, random_state=42,
                               stratify=df["label"])
train = train.sort_values("id").reset_index(drop=True)
test = test.sort_values("id").reset_index(drop=True)

answer = test[["id", "label"]]

d = PROBLEMS_DIR / "master"
# remove leftover image-track artifacts if any (old Fashion-MNIST setup)
for sub in ("train", "test"):
    if (d / sub).is_dir():
        for p in (d / sub).rglob("*"):
            if p.is_file():
                p.unlink()
        (d / sub).rmdir()
for f in ("labels.csv",):
    if (d / f).exists():
        (d / f).unlink()

d.mkdir(parents=True, exist_ok=True)
train.to_csv(d / "train.csv", index=False)
test[["id"] + FEATURES].to_csv(d / "test.csv", index=False)  # NO label — secret!
answer.to_csv(d / "answer.csv", index=False)
pd.DataFrame({"id": test["id"], "prediction": 0}).to_csv(
    d / "sample_submission.csv", index=False)

# ---- baseline: class-weighted HGB + threshold tuning ----
X, y = train[FEATURES], train["label"]
X_test = test[FEATURES]

# hold out 20% of train for threshold tuning
X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2,
                                            random_state=42, stratify=y)

clf = HistGradientBoostingClassifier(
    max_iter=300, learning_rate=0.05, max_depth=6,
    class_weight="balanced", random_state=42)
clf.fit(X_tr, y_tr)

proba_val = clf.predict_proba(X_val)[:, 1]
# pick the threshold maximizing validation F1 (positive class)
best_t, best_f1 = 0.5, 0.0
for t in np.arange(0.05, 0.95, 0.025):
    f1v = f1_score(y_val, (proba_val >= t).astype(int),
                   average="binary", pos_label=1, zero_division=0)
    if f1v > best_f1:
        best_t, best_f1 = t, f1v

proba_te = clf.predict_proba(X_test)[:, 1]
pred = (proba_te >= best_t).astype(int)
baseline = f1_score(test["label"], pred, average="binary", pos_label=1,
                    zero_division=0)

pd.DataFrame({"id": test["id"], "prediction": pred}).to_csv(
    "/tmp/master_submission.csv", index=False)

print(f"Train rows: {len(train)} | Test rows: {len(test)}")
print(f"Features: {len(FEATURES)} | positive rate (test): {test['label'].mean():.4f}")
print(f"Baseline (HGB class-weighted, threshold {best_t:.3f}) F1(pos): {baseline:.4f}")
print(f"Wrote {d / 'train.csv'} / test.csv / answer.csv / sample_submission.csv")
print("Baseline submission saved to /tmp/master_submission.csv")
