"""Utility: produce a quick demo submission for a given problem.
Usage: /arise/venv/bin/python make_demo_submission.py <explorer|innovator|master> [out.csv]
  - explorer / innovator: trains a LogisticRegression on the problem's train.csv
  - master: random labels (demo only — real participants use a CNN)
"""
import sys
import random
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backend"))
from config import PROBLEMS_DIR  # noqa: E402

problem = sys.argv[1] if len(sys.argv) > 1 else "explorer"
out = Path(sys.argv[2] if len(sys.argv) > 2 else f"/tmp/{problem}_submission.csv")

d = PROBLEMS_DIR / problem
test = pd.read_csv(d / "test.csv")

if problem == "master":
    n = len(test)
    random.seed(1)
    pred = [random.randint(0, 3) for _ in range(n)]
else:
    train = pd.read_csv(d / "train.csv")
    feats = [c for c in train.columns if c.startswith("f")]
    model = LogisticRegression(max_iter=2000)
    model.fit(train[feats], train["label"])
    pred = model.predict(test[feats])

sub = pd.DataFrame({"id": test["id"], "prediction": pred})
sub.to_csv(out, index=False)
print(f"Wrote {out} ({len(sub)} rows, problem={problem})")
