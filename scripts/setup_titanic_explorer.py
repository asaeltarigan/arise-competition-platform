"""ARISE 2026 — set up the EXPLORER problem with the Titanic dataset.

Downloads (or reuses) the classic Titanic CSV, does a stratified 80/20 split,
and writes:
  /arise/problems/explorer/train.csv   (id, features..., label)
  /arise/problems/explorer/test.csv    (id, features...)
  /arise/problems/explorer/answer.csv  (id, label)  <- SECRET, never distribute

Missing Age/Embarked values are LEFT as-is (participants learn preprocessing).
Also writes a LogisticRegression baseline submission for sanity checks.

Usage: /arise/venv/bin/python /arise/scripts/setup_titanic_explorer.py
"""
import sys
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backend"))
from config import PROBLEMS_DIR  # noqa: E402

URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
LOCAL = Path("/tmp/titanic.csv")

if not LOCAL.exists():
    import urllib.request
    print("Downloading titanic.csv...")
    urllib.request.urlretrieve(URL, LOCAL)

df = pd.read_csv(LOCAL)

FEATURES = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked",
            "name", "ticket", "cabin"]
TARGET = "survived"

data = df.rename(columns={"PassengerId": "id", "Survived": TARGET})[
    ["id", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked",
     "Name", "Ticket", "Cabin", TARGET]
].copy()
# Normalize to lowercase friendly names
data.columns = ["id"] + FEATURES + ["label"]

train, test = train_test_split(data, test_size=0.2, random_state=42,
                               stratify=data["label"])
train = train.sort_values("id").reset_index(drop=True)
test = test.sort_values("id").reset_index(drop=True)

answer = test[["id", "label"]]

d = PROBLEMS_DIR / "explorer"
d.mkdir(parents=True, exist_ok=True)
train.to_csv(d / "train.csv", index=False)
test[["id"] + FEATURES].to_csv(d / "test.csv", index=False)  # NO label column — it's the secret!
answer.to_csv(d / "answer.csv", index=False)

# Baseline: LR on the 7 core features only (median/mode imputation + one-hot).
CORE = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]
X = train[CORE].copy()
X["age"] = X["age"].fillna(X["age"].median())
X["embarked"] = X["embarked"].fillna(X["embarked"].mode()[0])
X = pd.get_dummies(X, columns=["sex", "embarked"])
X_test = test[CORE].copy()
X_test["age"] = X_test["age"].fillna(X_test["age"].median())
X_test["embarked"] = X_test["embarked"].fillna(X_test["embarked"].mode()[0])
X_test = pd.get_dummies(X_test, columns=["sex", "embarked"])
X_test = X_test.reindex(columns=X.columns, fill_value=0)

from sklearn.metrics import accuracy_score  # noqa: E402
lr = LogisticRegression(max_iter=1000)
lr.fit(X, train["label"])
pred = lr.predict(X_test)
baseline = accuracy_score(test["label"], pred)

sub = pd.DataFrame({"id": test["id"], "prediction": pred})
sub.to_csv("/tmp/explorer_submission.csv", index=False)

print(f"Train rows: {len(train)} | Test rows: {len(test)}")
print(f"Features: {FEATURES}")
print(f"Baseline (LogisticRegression) accuracy: {baseline:.4f}")
print(f"Wrote {d / 'train.csv'} / test.csv / answer.csv (answer is SECRET)")
print("Baseline submission saved to /tmp/explorer_submission.csv")
