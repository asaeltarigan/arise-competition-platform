"""ARISE 2026 — CSV validation (per guidebook rules)."""
import pandas as pd

from config import ID_COL, PRED_COL

class ValidationError(Exception):
    pass


def validate_submission(file_path, filename, test_csv):
    """Raise ValidationError with a human-readable reason if invalid.
    Returns the number of rows on success.

    NOTE: the uploaded file's NAME is intentionally NOT enforced — any
    filename is accepted (the content is validated instead).
    """
    try:
        df = pd.read_csv(file_path)
    except Exception as e:  # noqa: BLE001
        raise ValidationError(f"Cannot parse CSV: {e}")

    expected_cols = [ID_COL, PRED_COL]
    if list(df.columns) != expected_cols:
        raise ValidationError(
            f"Columns must be exactly [{', '.join(expected_cols)}]; got {list(df.columns)}")

    if df[ID_COL].duplicated().any():
        raise ValidationError("Duplicate ids found in submission")

    if df[ID_COL].isna().any() or df[PRED_COL].isna().any():
        raise ValidationError("Missing (NaN) values found in id or prediction")

    try:
        test = pd.read_csv(test_csv)
    except FileNotFoundError:
        raise ValidationError("Server error: test.csv not found (contact organizer)")

    if len(df) != len(test):
        raise ValidationError(
            f"Row count {len(df)} does not match test.csv ({len(test)})")

    test_ids = set(test[ID_COL].astype(str))
    sub_ids = set(df[ID_COL].astype(str))
    if sub_ids != test_ids:
        missing = test_ids - sub_ids
        extra = sub_ids - test_ids
        bits = []
        if missing:
            bits.append(f"{len(missing)} missing id(s)")
        if extra:
            bits.append(f"{len(extra)} unknown id(s)")
        raise ValidationError("ID mismatch: " + ", ".join(bits))

    return len(df)
