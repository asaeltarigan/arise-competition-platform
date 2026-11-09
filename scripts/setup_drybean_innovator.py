"""ARISE 2026 — set up the INNOVATOR problem with the UCI Dry Bean dataset.

Downloads the UCI Dry Bean dataset (13,611 rows, 16 numeric features, 7 classes),
does a stratified 80/20 split, and writes:
  /arise/problems/innovator/train.csv   (id, features..., label)
  /arise/problems/innovator/test.csv    (id, features...)
  /arise/problems/innovator/answer.csv  (id, label)  <- SECRET, never distribute
  /arise/problems/innovator/sample_submission.csv

Features keep their real names (educational); classes are mapped to 0..6.
Also writes a RandomForest baseline submission for sanity checks.

Usage: /arise/venv/bin/python /arise/scripts/setup_drybean_innovator.py
"""
import sys
import zipfile
import urllib.request
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backend"))
from config import PROBLEMS_DIR  # noqa: E402

URL = "https://archive.ics.uci.edu/static/public/602/dry+bean+dataset.zip"
LOCAL = Path("/tmp/drybean.zip")

if not LOCAL.exists():
    print("Downloading Dry Bean dataset...")
    urllib.request.urlretrieve(URL, LOCAL)

# Inspect what's inside the zip, parse the ARFF (xlsx engine is unreliable here)
with zipfile.ZipFile(LOCAL) as z:
    names = z.namelist()
    print("zip entries:", names)
    member = next(n for n in names if n.endswith(".arff"))
    tmp_arff = Path("/tmp") / Path(member).name
    with z.open(member) as src, open(tmp_arff, "wb") as dst:
        dst.write(src.read())

# Parse ARFF manually (simple structure: @ATTRIBUTE lines then @DATA rows)
attrs, rows = [], []
with open(tmp_arff, "r") as f:
    in_data = False
    for line in f:
        line = line.strip()
        if not line or line.startswith("%"):
            continue
        if line.upper().startswith("@ATTRIBUTE"):
            name = line.split()[1]
            attrs.append(name)
        elif line.upper().startswith("@DATA"):
            in_data = True
        elif in_data:
            rows.append(line.split(","))

df = pd.DataFrame(rows, columns=attrs)
# numeric coercion (Class stays object)
for c in attrs:
    if c != "Class":
        df[c] = pd.to_numeric(df[c])
print("raw shape:", df.shape)
print("columns:", list(df.columns))

# Class column is the only string column (pandas 3.x uses StringDtype)
CLASS_COL = [c for c in df.columns if pd.api.types.is_string_dtype(df[c])]
assert len(CLASS_COL) == 1, f"expected exactly one string column, got {CLASS_COL}"
CLASS_COL = CLASS_COL[0]

FEATURES = [c for c in df.columns if c != CLASS_COL]
classes = sorted(df[CLASS_COL].unique())
print("classes:", classes)
class_map = {c: i for i, c in enumerate(classes)}

data = df.rename(columns={CLASS_COL: "label"}).copy()
data["label"] = data["label"].map(class_map)
data.insert(0, "id", range(1, len(data) + 1))
data = data[["id"] + FEATURES + ["label"]]

train, test = train_test_split(data, test_size=0.2, random_state=42,
                               stratify=data["label"])
train = train.sort_values("id").reset_index(drop=True)
test = test.sort_values("id").reset_index(drop=True)

answer = test[["id", "label"]]

d = PROBLEMS_DIR / "innovator"
d.mkdir(parents=True, exist_ok=True)
train.to_csv(d / "train.csv", index=False)
test[["id"] + FEATURES].to_csv(d / "test.csv", index=False)  # NO label — secret!
answer.to_csv(d / "answer.csv", index=False)
# Sample submission: valid format, majority class (0) as 'dumb' baseline.
sample = pd.DataFrame({"id": test["id"], "prediction": 0})
sample.to_csv(d / "sample_submission.csv", index=False)

# Baseline: RandomForest (guidebook: Innovator = ensembling).
X, y = train[FEATURES], train["label"]
rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(X, y)
pred = rf.predict(test[FEATURES])
baseline = accuracy_score(test["label"], pred)

sub = pd.DataFrame({"id": test["id"], "prediction": pred})
sub.to_csv("/tmp/innovator_submission.csv", index=False)

print(f"Train rows: {len(train)} | Test rows: {len(test)}")
print(f"Features: {len(FEATURES)} | Classes: {len(classes)} (mapped 0..{len(classes)-1})")
print(f"Baseline (RandomForest) accuracy: {baseline:.4f}")
print(f"Wrote {d / 'train.csv'} / test.csv / answer.csv / sample_submission.csv")
print("Baseline submission saved to /tmp/innovator_submission.csv")
