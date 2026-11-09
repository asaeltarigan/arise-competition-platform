#!/usr/bin/env python3
"""Build the ARISE starter notebooks (skeleton editions) for Explorer + Innovator.

Each notebook is a *guided competition skeleton*: working scaffolding with
strategic TODOs that force understanding (drop-column decisions, imputation
choices, feature engineering, model comparison, validation, experiment log).
Not fill-in-the-syntax — fill-in-the-understanding.

Output:
  /arise/frontend/notebooks/ARISE_Explorer_Titanic_Starter.ipynb
  /arise/frontend/notebooks/ARISE_Innovator_DryBean_Starter.ipynb
"""
import json
from pathlib import Path

OUT = Path("/arise/frontend/notebooks")
OUT.mkdir(parents=True, exist_ok=True)


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(True)}


def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": src.splitlines(True)}


def notebook(cells, title="ARISE Starter"):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


# =====================================================================
# EXPLORER — Titanic
# =====================================================================
explorer = [
    md("""# 🟢 ARISE Explorer — Titanic Starter Notebook

Welcome to the **Explorer** level of ARISE 2026!

Your job: build a classifier that predicts whether a Titanic passenger survived.
This notebook is a **guided skeleton** — some parts are done for you, but the
parts that test your *understanding* are left as TODOs. That's the point.

## The rules of this notebook

- You **may** use AI assistants (ChatGPT, Gemini, Claude...). Explain everything
  you do in your own words.
- **Do NOT change the train/test split.** Use exactly the `train.csv` and
  `test.csv` provided (same files every participant gets).
- Submit `submission.csv` on the ARISE portal (max 5/day, best score counts).

## How this notebook works

1. Load the official train/test files
2. Explore the data (EDA)
3. **TODO** — decide what to drop and why
4. **TODO** — choose your missing-value strategy
5. **TODO** — engineer at least 2 features
6. Build a preprocessing pipeline
7. **TODO** — compare at least 3 models
8. **TODO** — keep an experiment log
9. Predict + save `submission.csv`

> The baseline (simple model, no feature engineering) scores about **0.79**.
> Your goal: **beat it** — 0.82 is good, 0.85+ is excellent.
"""),
    md("""## 1. Setup — install & import"""),
    code("""# If you run this in Google Colab, install nothing extra for this notebook.
# (sklearn + pandas are pre-installed.)
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

print("Setup complete!")"""),
    md("""## 2. Load the official data

These are the **official ARISE files** — use exactly these, do not re-split.
`train.csv` has the target column `label`; `test.csv` has **no** label (that's
the secret — the server holds the answers)."""),
    code("""# Put the downloaded ARISE Explorer files in the same folder as this notebook.
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

print("Train shape:", train_df.shape)
print("Test shape :", test_df.shape)
print("\\nTrain columns:", train_df.columns.tolist())"""),
    md("""## 3. EDA — look before you leap"""),
    code("""train_df.head()"""),
    code("""print("Missing values (train):")
print(train_df.isnull().sum())
print("\\nTarget distribution:")
print(train_df["label"].value_counts(normalize=True))"""),
    md("""## 4. TODO — what should we drop? 🎯

Not every column is a useful model feature. Look at `train_df` columns above
and decide which ones should **not** be fed directly to a model.

Fill in the list, then **justify each choice in the markdown cell below**."""),
    code("""# ============================================================
# YOUR DECISION — which columns to drop?
# ============================================================
DROP_COLUMNS = [
    # TODO: add at least 2 column names (e.g. "Name", "Ticket", "Cabin" ...)
]"""),
    md("""### Your justification (write at least 2 sentences)

> TODO: Explain *why* each dropped column is not a good direct feature.
> Hints to consider: is it an identifier? Is it mostly missing? Is it a
> free-text field that needs engineering before it becomes useful?"""),
    md("""## 5. TODO — missing values strategy 🎯

`Age` and `Embarked` have missing values (you saw it above). Choose a strategy
for each and **explain your choice**. The `SimpleImputer` options are:
`"mean"`, `"median"`, `"most_frequent"` (and `"constant"`)."""),
    code("""# ============================================================
# YOUR DECISION — imputation strategies
# ============================================================
AGE_IMPUTATION = "________"        # TODO: pick one
EMBARKED_IMPUTATION = "________"   # TODO: pick one"""),
    md("""### Justify your imputation choices (2+ sentences)

> TODO: Why that strategy for Age? Why that strategy for Embarked?
> What happens if you choose the wrong one? (Think: outliers in Age,
> skew in Embarked.)"""),
    md("""## 6. TODO — feature engineering 🎯

This is where Titanic gets interesting. Add **at least 2 new features**.
Do not simply duplicate existing columns. Ideas you may use (or invent your own):

- `FamilySize = SibSp + Parch + 1`
- `IsAlone` (1 if FamilySize == 1 else 0)
- `Title` extracted from the `Name` column (Mr, Mrs, Miss, Master...)
- Fare per person (`Fare / FamilySize`)

The assert at the end is **enforced** — your notebook must produce 2+ new features."""),
    code("""def create_features(df):
    \"\"\"Add engineered features. Returns a copy with new columns added.\"\"\"
    df = df.copy()

    # TODO 1: FamilySize = SibSp + Parch + 1
    # TODO 2: IsAlone (1 if alone, else 0)
    # TODO 3: Title extracted from Name (or another feature of your own)

    return df


# Build feature engineering into both train and test
train_fe = create_features(train_df)
test_fe = create_features(test_df)

# ---- ENFORCED: at least 2 new features ----
new_features = [c for c in train_fe.columns if c not in train_df.columns]
assert len(new_features) >= 2, f"You need at least 2 new features, got {new_features}"
print("New features created:", new_features)"""),
    md("""## 7. Preprocessing pipeline

Now assemble the pipeline. It should:
1. Drop the columns you chose (keep them out of the features)
2. Impute missing values with your chosen strategies
3. One-hot encode categorical columns
4. Scale numeric columns

Fill the gaps marked `________`."""),
    code("""TARGET = "label"

# Features to keep = everything except target + your dropped columns
feature_cols = [c for c in train_fe.columns if c not in [TARGET] + DROP_COLUMNS]

# Split numeric vs categorical (adjust to your engineered features!)
numeric_features = [c for c in feature_cols
                    if train_fe[c].dtype in ("int64", "float64")]
categorical_features = [c for c in feature_cols
                        if c not in numeric_features]

print("Numeric:", numeric_features)
print("Categorical:", categorical_features)"""),
    code("""numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy=AGE_IMPUTATION)),
    ("scaler", StandardScaler()),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy=EMBARKED_IMPUTATION)),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features),
])"""),
    md("""## 8. TODO — compare at least 3 models 🎯

Train **at least 3 different algorithms** with the same 5-fold cross-validation
and compare them. You get the scaffolding — fill in your model choices."""),
    code("""# ============================================================
# YOUR DECISION — at least 3 models to compare
# ============================================================
MODELS = {
    # TODO: model name -> estimator instance
    # e.g. "Logistic Regression": LogisticRegression(max_iter=1000),
    #      "Random Forest":       RandomForestClassifier(random_state=42),
    #      "Gradient Boosting":   GradientBoostingClassifier(random_state=42),
}

X = train_fe[feature_cols]
y = train_fe[TARGET]

results = []
for name, model in MODELS.items():
    pipe = Pipeline([("preprocessor", preprocessor), ("model", model)])
    cv = cross_val_score(pipe, X, y, cv=5, scoring="accuracy")
    results.append({"Model": name, "CV accuracy": round(cv.mean(), 4),
                    "±std": round(cv.std(), 4)})
    print(f"{name:>22}: {cv.mean():.4f} ± {cv.std():.4f}")

results_df = pd.DataFrame(results).sort_values("CV accuracy", ascending=False)
results_df"""),
    md("""### Which model won? Why do you think that is?

> TODO: 2+ sentences on why the best model won. Consider the data size,
> feature types, and what each algorithm assumes."""),
    md("""## 9. Train the winner + predict

Pick your best model (or an ensemble of the best few) and generate the
submission file."""),
    code("""# ============================================================
# YOUR DECISION — final model
# ============================================================
FINAL_MODEL = RandomForestClassifier(random_state=42)  # TODO: your best model

final_pipe = Pipeline([("preprocessor", preprocessor), ("model", FINAL_MODEL)])
final_pipe.fit(X, y)

# Predict the OFFICIAL test set (no label — that's the point!)
prediction = final_pipe.predict(test_fe[feature_cols])

submission = pd.DataFrame({"id": test_df["id"], "prediction": prediction})
submission.to_csv("submission.csv", index=False)
print("submission.csv created!", submission.shape)"""),
    md("""## 10. Self-check (always run before uploading)"""),
    code("""assert list(submission.columns) == ["id", "prediction"]
assert len(submission) == len(test_df)
assert submission["id"].equals(test_df["id"])
assert submission["prediction"].notna().all()
print("✓ Correct columns, row count, ids, no missing predictions")
print("✓ Ready to upload submission.csv")"""),
    md("""## 11. Experiment log 🧪 (fill as you go)

| # | Experiment (change you made) | CV accuracy | Did it help? |
|---|---|---|---|
| 1 | Baseline (no engineering) | ~0.79 | — |
| 2 | + FamilySize |  |  |
| 3 | + Title extraction |  |  |
| 4 | + Different model |  |  |
| 5 | + Hyperparameter tuning |  |  |

**What did you learn?** (2+ sentences)

> TODO: write what you tried, what worked, what didn't, and why.

---

## 🏁 The challenge

| Score | Level |
|---|---|
| 0.79 | Baseline (everyone can) |
| 0.82 | Good — real feature engineering |
| 0.85+ | Excellent — tuning + ensembling |

**Remember:** your leaderboard score is only 15% of the final mark. The
presentation & Q&A (70%) is where you prove you actually understand what you
did. The experiment log above is your evidence. Good luck! 🚢🔥"""),
]

# =====================================================================
# INNOVATOR — Dry Bean
# =====================================================================
innovator = [
    md("""# 🔵 ARISE Innovator — Dry Bean Starter Notebook

Welcome to the **Innovator** level of ARISE 2026!

Your job: classify images of dry beans into **7 varieties** using their
measured geometric features. This level is *deliberately harder* than Explorer
— it's about **methodology**, not just running a model.

## The rules

- You **may** use AI assistants. You must be able to explain everything.
- Use exactly `train.csv` / `test.csv` provided. Never re-split.
- Submit `submission.csv` (5/day max, best score counts).

## What "good" looks like here

- The data is **multiclass** (7 classes), **imbalanced**, and has **many
  correlated features**.
- A plain Random Forest gets ~**0.91**. To stand out you'll need feature
  selection, scaling-aware choices, hyperparameter tuning, and ensembling.
- A score of **0.93+** is good; **0.94+** is excellent.

## Your workflow

1. Load data + EDA (class balance, distributions, correlations)
2. **TODO** — feature selection (justify!)
3. **TODO** — compare at least 4 models
4. **TODO** — hyperparameter tuning
5. **TODO** — error analysis (which classes get confused?)
6. **TODO** — experiment log
7. Predict + submit
"""),
    md("""## 1. Setup"""),
    code("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

import warnings
warnings.filterwarnings("ignore")
print("Setup complete!")"""),
    md("""## 2. Load the official data"""),
    code("""train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

print("Train shape:", train_df.shape)
print("Test shape :", test_df.shape)
print("\\nLabel distribution (train):")
print(train_df["label"].value_counts(normalize=True))"""),
    md("""## 3. EDA — distributions & correlations

Run these, then **answer the questions** below in your own words."""),
    code("""train_df.describe().T"""),
    code("""# Class balance
sns.countplot(data=train_df, x="label")
plt.xticks(rotation=45)
plt.title("Class distribution (train)")
plt.show()"""),
    code("""# Correlation between numeric features
feat_cols = [c for c in train_df.columns if c not in ("id", "label")]
corr = train_df[feat_cols].corr()
plt.figure(figsize=(12, 9))
sns.heatmap(corr, cmap="coolwarm", center=0, annot=False)
plt.title("Feature correlation heatmap")
plt.show()

# Highly correlated pairs
high = [(a, b) for a in corr.columns for b in corr.columns
        if a < b and abs(corr[a][b]) > 0.9]
print("Highly correlated pairs (>0.9):")
for a, b in high[:10]:
    print(f"  {a} <-> {b}: {corr[a][b]:.2f}")"""),
    md("""### EDA observations (write 3+ sentences)

> TODO: What did you learn?
> 1. Which classes are rare / common (imbalance)?
> 2. Which features are strongly correlated (redundant)?
> 3. Which features look most useful for separating classes?"""),
    md("""## 4. TODO — feature selection 🎯

The dataset has 16+ features and several are highly correlated. Decide which
features you'll keep, then **justify** your choice. You may also experiment
with selection methods (e.g. tree feature importance) — but your final
`SELECTED_FEATURES` is what the pipeline uses."""),
    code("""# ============================================================
# YOUR DECISION — feature selection
# ============================================================
SELECTED_FEATURES = [
    # TODO: pick the features you believe are most useful
    # (start from feat_cols, then remove correlated/weak ones)
]"""),
    md("""### Justify your feature selection (3+ sentences)

> TODO: Why drop the features you dropped? Are they redundant (correlated)?
> Do they help or hurt the model? How did you test this?"""),
    md("""## 5. Preprocessing

Dry Bean features have **very different scales** (Area ~ 10^4, Roundness ~ 1).
Scaling matters for some models (LR, SVM, KNN) and not others (trees).
Fill in the pipeline."""),
    code("""TARGET = "label"

X = train_df[SELECTED_FEATURES]
y = train_df[TARGET]
X_test = test_df[SELECTED_FEATURES]

# NOTE: scaling is inside the pipeline so CV stays honest
preprocessor = StandardScaler()"""),
    md("""## 6. TODO — compare at least 4 models 🎯

Compare **at least 4** algorithms using the same 5-fold CV. Report accuracy,
**macro F1** (important for imbalanced multiclass), and training time."""),
    code("""import time

# ============================================================
# YOUR DECISION — at least 4 models to compare
# ============================================================
MODELS = {
    # TODO: add at least 4
    # e.g. "Logistic Regression":  LogisticRegression(max_iter=2000),
    #      "Random Forest":        RandomForestClassifier(random_state=42),
    #      "Gradient Boosting":    GradientBoostingClassifier(random_state=42),
    #      "SVM (RBF)":            SVC(random_state=42),
}

results = []
for name, model in MODELS.items():
    pipe = Pipeline([("scaler", preprocessor), ("model", model)])
    t0 = time.time()
    cv_acc = cross_val_score(pipe, X, y, cv=5, scoring="accuracy")
    cv_f1 = cross_val_score(pipe, X, y, cv=5,
                            scoring="f1_macro")
    results.append({"Model": name,
                    "CV accuracy": round(cv_acc.mean(), 4),
                    "CV macro F1": round(cv_f1.mean(), 4),
                    "time (s)": round(time.time() - t0, 2)})
    print(f"{name:>22}: acc={cv_acc.mean():.4f}  f1={cv_f1.mean():.4f}")

results_df = pd.DataFrame(results).sort_values("CV macro F1", ascending=False)
results_df"""),
    md("""### Model comparison takeaways (2+ sentences)

> TODO: Which model(s) performed best? Why? Where does scaling matter most?
> What trade-off between accuracy, F1, and training time do you see?"""),
    md("""## 7. TODO — hyperparameter tuning 🎯

Take your best model and tune it with `GridSearchCV` (or RandomizedSearchCV).
You choose the search space — that's the point."""),
    code("""# ============================================================
# YOUR DECISION — search space for your best model
# ============================================================
BEST_MODEL = RandomForestClassifier(random_state=42)  # TODO: your best

PARAM_GRID = {
    # TODO: define the search space
    # e.g. "n_estimators": [100, 200, 300],
    #      "max_depth": [None, 10, 20],
}"""),
    code("""search = GridSearchCV(BEST_MODEL, PARAM_GRID, cv=5,
                      scoring="f1_macro", n_jobs=-1, verbose=1)
search.fit(X, y)

print("Best params:", search.best_params_)
print("Best CV macro F1:", round(search.best_score_, 4))"""),
    md("""### Tuning results (2+ sentences)

> TODO: What hyperparameters mattered most? Did tuning beat the untuned model?"""),
    md("""## 8. TODO — error analysis 🎯 (this is the "great" filter)

Even good models confuse *some* classes. Fit your best model and examine the
confusion matrix. **Which two classes get confused most often, and why do you
think that is?**"""),
    code("""from sklearn.model_selection import train_test_split as tts

# Hold out 20% as a validation set for this analysis (same seed = reproducible)
X_tr, X_va, y_tr, y_va = tts(X, y, test_size=0.2, random_state=42, stratify=y)

final = Pipeline([("scaler", preprocessor), ("model", search.best_estimator_)])
final.fit(X_tr, y_tr)
y_pred_va = final.predict(X_va)

# Confusion matrix
cm = confusion_matrix(y_va, y_pred_va)
labels = sorted(y.unique())
plt.figure(figsize=(9, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion matrix (validation)")
plt.show()

print(classification_report(y_va, y_pred_va))"""),
    md("""### Error analysis (3+ sentences)

> TODO: Which two classes does your model confuse most? Look at their features
> (size, shape) — why are they hard to tell apart? How might you reduce this?"""),
    md("""## 9. Final model + submission

Train your tuned best model on **all** training data and predict the official test set."""),
    code("""final.fit(X, y)
prediction = final.predict(X_test)

submission = pd.DataFrame({"id": test_df["id"], "prediction": prediction})
submission.to_csv("submission.csv", index=False)
print("submission.csv created!", submission.shape)"""),
    md("""## 10. Self-check"""),
    code("""assert list(submission.columns) == ["id", "prediction"]
assert len(submission) == len(test_df)
assert submission["id"].equals(test_df["id"])
assert submission["prediction"].notna().all()
print("✓ Ready to upload submission.csv")"""),
    md("""## 11. Experiment log 🧪

| # | Experiment | CV macro F1 | Result |
|---|---|---|---|
| 1 | Random Forest baseline (all features) | ~0.91 | Baseline |
| 2 | Feature selection |  |  |
| 3 | Scaling + model comparison |  |  |
| 4 | Hyperparameter tuning |  |  |
| 5 | Ensemble / ExtraTrees / etc. |  |  |

**What did you learn?** (3+ sentences)

> TODO: your honest reflection on what moved the needle and why.

---

## 🏁 The challenge

| Score | Level |
|---|---|
| 0.91 | Baseline (RF, all features) |
| 0.93+ | Good — selection + tuning |
| 0.94+ | Excellent — ensembling + deep tuning |

Your leaderboard score is 15% of the final mark; presentation + Q&A is the rest.
The experiment log is your evidence. Good luck! 🌱🔥"""),
]

Path(OUT / "ARISE_Explorer_Titanic_Starter.ipynb").write_text(
    json.dumps(notebook(explorer, "ARISE Explorer Starter"), indent=1))
Path(OUT / "ARISE_Innovator_DryBean_Starter.ipynb").write_text(
    json.dumps(notebook(innovator, "ARISE Innovator Starter"), indent=1))
print("Wrote:")
print(" ", OUT / "ARISE_Explorer_Titanic_Starter.ipynb")
print(" ", OUT / "ARISE_Innovator_DryBean_Starter.ipynb")
