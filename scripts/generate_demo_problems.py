"""Generate DEMO datasets for all three ARISE 2026 problems.

Creates:
  /arise/problems/explorer/{train,test,answer}.csv   — easy tabular (LR ~0.90)
  /arise/problems/innovator/{train,test,answer}.csv  — harder tabular (LR ~0.70, RF better)
  /arise/problems/master/                            — synthetic image classification
      train/{img0001..}.jpg, test/{img0101..}.jpg, labels.csv, answer.csv

Replace any problem's files with the real ARISE data later — the system only
needs train.csv / test.csv / answer.csv (id + label) per problem folder.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backend"))
from config import PROBLEMS_DIR  # noqa: E402


def write_tabular(problem, n_features, noise, n_train, n_test, flip, seed):
    d = PROBLEMS_DIR / problem
    d.mkdir(parents=True, exist_ok=True)
    n_informative = max(2, n_features - 2)
    n_redundant = 1 if n_features >= 4 else 0
    X, y = make_classification(
        n_samples=n_train + n_test, n_features=n_features,
        n_informative=n_informative, n_redundant=n_redundant,
        n_clusters_per_class=1, flip_y=flip, random_state=seed,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=n_test, random_state=seed + 1, stratify=y)
    feats = [f"f{i+1}" for i in range(n_features)]

    train = pd.DataFrame(X_train, columns=feats)
    train["label"] = y_train
    train.insert(0, "id", range(1, n_train + 1))
    test = pd.DataFrame(X_test, columns=feats)
    test.insert(0, "id", range(1, n_test + 1))
    answer = pd.DataFrame({"id": range(1, n_test + 1), "label": y_test})

    train.to_csv(d / "train.csv", index=False)
    test.to_csv(d / "test.csv", index=False)
    answer.to_csv(d / "answer.csv", index=False)

    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    from sklearn.metrics import accuracy_score  # noqa: E402
    print(f"[{problem}] rows={n_train} train / {n_test} test  "
          f"LR baseline={accuracy_score(y_test, lr.predict(X_test)):.3f}  -> {d}")


def draw_shape(draw, shape, box, color):
    if shape == 0:      # circle
        draw.ellipse(box, fill=color)
    elif shape == 1:    # square
        draw.rectangle(box, fill=color)
    elif shape == 2:    # triangle
        x0, y0, x1, y1 = box
        draw.polygon([(x0, y1), ((x0 + x1) // 2, y0), (x1, y1)], fill=color)
    else:               # cross
        cx, cy = (box[0] + box[2]) // 2, (box[1] + box[3]) // 2
        r = (box[2] - box[0]) // 2
        draw.rectangle([cx - r // 2, cy - r, cx + r // 2, cy + r], fill=color)
        draw.rectangle([cx - r, cy - r // 2, cx + r, cy + r // 2], fill=color)


def write_master(n_train=400, n_test=120, size=16, seed=42):
    d = PROBLEMS_DIR / "master"
    (d / "train").mkdir(parents=True, exist_ok=True)
    (d / "test").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    def make_image(shape, n):
        img = Image.new("L", (size, size), 0)
        dr = ImageDraw.Draw(img)
        pad = 2
        box = [pad, pad, size - 1 - pad, size - 1 - pad]
        draw_shape(dr, shape, box, 255)
        # add noise so it is not trivially memorized
        arr = np.array(img, dtype=np.int16)
        arr += rng.integers(-35, 36, arr.shape)
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)

    rows, labels = [], []
    for i in range(n_train + n_test):
        shape = i % 4
        im = make_image(shape, i)
        if i < n_train:
            fname = f"img{i+1:04d}.jpg"
            im.save(d / "train" / fname, "JPEG")
            rows.append(fname)
            labels.append(shape)
        else:
            fname = f"img{i+1:04d}.jpg"
            im.save(d / "test" / fname, "JPEG")
            rows.append(fname)
            labels.append(shape)

    train_lbl = pd.DataFrame({"id": rows[:n_train], "label": labels[:n_train]})
    train_lbl.to_csv(d / "labels.csv", index=False)
    test_ids = pd.DataFrame({"id": rows[n_train:]})
    test_ids.to_csv(d / "test.csv", index=False)
    answer = pd.DataFrame({"id": rows[n_train:], "label": labels[n_train:]})
    answer.to_csv(d / "answer.csv", index=False)
    print(f"[master] {n_train} train images / {n_test} test images (4 classes) -> {d}")


if __name__ == "__main__":
    write_tabular("explorer", n_features=3, noise=0, n_train=800, n_test=200,
                  flip=0.05, seed=42)
    write_tabular("innovator", n_features=10, noise=0, n_train=1200, n_test=300,
                  flip=0.25, seed=7)
    write_master()
    print("Done. Replace any problem folder with real ARISE data when ready.")
