"""Offline tests; no medical dataset download is required."""
from pathlib import Path
import json

import numpy as np
import pandas as pd

from preprocess import run


def fake_heart(rows: int = 80) -> pd.DataFrame:
    rng = np.random.default_rng(23)
    data = {
        "age": rng.integers(30, 85, rows),
        "trestbps": rng.integers(95, 185, rows),
        "chol": rng.integers(130, 380, rows),
        "thalach": rng.integers(70, 195, rows),
        "oldpeak": rng.uniform(0, 4, rows),
        "ca": rng.choice([0, 1, 2, 3, "?"], rows),
        "sex": rng.choice([0, 1], rows),
        "cp": rng.choice([1, 2, 3, 4], rows),
        "fbs": rng.choice([0, 1], rows),
        "restecg": rng.choice([0, 1, 2], rows),
        "exang": rng.choice([0, 1], rows),
        "slope": rng.choice([1, 2, 3], rows),
        "thal": rng.choice([3, 6, 7, "?"], rows),
        "num": rng.integers(0, 5, rows),
    }
    return pd.DataFrame(data)


def test_heart_pipeline(tmp_path: Path) -> None:
    source = tmp_path / "heart.csv"
    fake_heart().to_csv(source, index=False)
    folder = run(csv_path=str(source), sample_size=60, query_count=10,
                 pad_to=32, top_k=5, output_dir=tmp_path / "out")
    db = np.load(folder / "database.npy")
    qs = np.load(folder / "queries.npy")
    assert db.shape == (50, 32)
    assert qs.shape == (10, 32)
    assert np.allclose(np.linalg.norm(db, axis=1), 1.0)
    assert np.allclose(np.linalg.norm(qs, axis=1), 1.0)
    assert len(pd.read_csv(folder / "plaintext_topk.csv")) == 50
    meta = json.loads((folder / "metadata.json").read_text())
    assert meta["dataset"] == "heart"
    assert not meta["outcome_label_used_as_feature"]


def test_new_query_uses_same_transform(tmp_path: Path) -> None:
    from preprocess import pad_and_normalize, prepare_columns
    import joblib

    source = tmp_path / "heart.csv"
    raw = fake_heart()
    raw.to_csv(source, index=False)
    folder = run(csv_path=str(source), query_count=10, output_dir=tmp_path / "out")
    transformer = joblib.load(folder / "preprocessor.joblib")
    new = raw.iloc[:1].copy()
    vector = pad_and_normalize(transformer.transform(prepare_columns(new)), 32)
    assert vector.shape == (1, 32)
    assert np.allclose(np.linalg.norm(vector, axis=1), 1.0)
