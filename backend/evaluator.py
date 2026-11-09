"""ARISE 2026 — evaluation engine (per-problem: metric + answer key)."""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error

from config import ID_COL, PRED_COL


def _align_labels(y_true, y_pred):
    """Normalize label dtype before sklearn scoring.

    Scania-style problems carry string labels ('neg'/'pos') while students may
    submit numeric predictions (0/1). sklearn refuses to mix label types and
    raises "Mix of label input types (string and number)" -> 500 on /submit.
    Convention: numeric 0..k maps to the k+1 classes in sorted order
    (0 -> first class alphabetically), which matches the 'neg'/'pos' case.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    t_str = y_true.dtype.kind in "OUS"
    p_str = y_pred.dtype.kind in "OUS"
    if t_str == p_str:
        return y_true, y_pred
    if t_str and not p_str:  # true=strings, pred=numbers -> map pred to strings
        classes = sorted(set(y_true))
        mapping = {int(i): c for i, c in enumerate(classes)}
        y_pred = np.array([mapping.get(int(v), str(int(v))) for v in y_pred])
    else:  # pred=strings, true=numbers -> map true to strings
        classes = sorted(set(y_pred))
        mapping = {int(i): c for i, c in enumerate(classes)}
        y_true = np.array([mapping.get(int(v), str(int(v))) for v in y_true])
    return y_true, y_pred


def evaluate(file_path, answer_csv, metric="accuracy"):
    """Score a validated submission against the answer key for one problem.
    Returns a float score (higher is better).
    """
    answer = pd.read_csv(answer_csv)
    # The ground-truth column is whatever comes after 'id' in answer.csv
    # (keeps config simple: answer_label is set per problem in config.PROBLEMS).
    label_col = [c for c in answer.columns if c != ID_COL][0]
    answer = answer.rename(columns={label_col: "y_true"})
    sub = pd.read_csv(file_path).rename(columns={PRED_COL: "y_pred"})
    merged = answer.merge(sub, on=ID_COL)
    y_true = merged["y_true"]
    y_pred = merged["y_pred"]

    if metric in ("accuracy", "f1", "f1_pos"):
        y_true, y_pred = _align_labels(y_true, y_pred)

    if metric == "accuracy":
        score = accuracy_score(y_true, y_pred)
    elif metric == "f1":
        score = f1_score(y_true, y_pred, average="macro", zero_division=0)
    elif metric == "f1_pos":
        # F1 of the positive class — standard for imbalanced rare-event tasks
        # like Scania APS (1.7% positive). Supports both int (1) and string
        # ("pos") label conventions.
        pos_label = "pos" if "pos" in set(y_true) else 1
        score = f1_score(y_true, y_pred, average="binary", pos_label=pos_label,
                         zero_division=0)
    elif metric == "rmse":
        # negated so higher is still better on the leaderboard
        score = -mean_squared_error(y_true, y_pred, squared=False)
    else:
        raise ValueError(f"Unknown metric: {metric}")

    return round(float(score), 6)
