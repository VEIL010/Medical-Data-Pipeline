### main 전처리 코드
from __future__ import annotations

import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, normalize

UCI_ID = 45
NUMERIC_COLUMNS = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
CATEGORICAL_COLUMNS = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]
INPUT_COLUMNS = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS
EXCLUDED_COLUMNS = ["num", "target"]




def load_dataset(csv_path: str | None) -> pd.DataFrame:
    """Load local CSV or fetch UCI Heart Disease features (target excluded)."""
    if csv_path:
        path = Path(csv_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"CSV file not found: {path}")
        return pd.read_csv(path, low_memory=False)

    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError as exc:
        raise SystemExit("Install dependencies: python -m pip install -r requirements.txt") from exc

    print("Downloading UCI Heart Disease (ID 45) ...")
    return fetch_ucirepo(id=UCI_ID).data.features.copy()




def prepare_columns(raw: pd.DataFrame) -> pd.DataFrame:
    """Exclude outcomes, enforce medical feature types and mark missing values."""
    frame = raw.copy()
    frame.columns = frame.columns.astype(str).str.strip()
    frame = frame.replace(r"^\s*\?\s*$", np.nan, regex=True)
    frame = frame.replace(r"^\s*$", np.nan, regex=True)
    frame = frame.drop(columns=EXCLUDED_COLUMNS, errors="ignore")

    missing = [column for column in INPUT_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing Heart Disease columns: {missing}")
    frame = frame.loc[:, INPUT_COLUMNS].copy()

    for column in NUMERIC_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for column in CATEGORICAL_COLUMNS:
        frame[column] = frame[column].astype("string").fillna("__MISSING__").astype(str)
    return frame


def make_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float64)),
    ])
    return ColumnTransformer([
        ("numeric", numeric, NUMERIC_COLUMNS),
        ("categorical", categorical, CATEGORICAL_COLUMNS),
    ], remainder="drop", sparse_threshold=0, verbose_feature_names_out=False)


def pad_and_normalize(vectors: np.ndarray, final_dimension: int) -> np.ndarray:
    if vectors.ndim != 2 or not len(vectors):
        raise ValueError("Expected a nonempty 2D matrix")
    if vectors.shape[1] > final_dimension:
        raise ValueError(
            f"Data has {vectors.shape[1]} encoded features; --pad-to {final_dimension} "
            "is too small. Try --pad-to 64 or omit it for automatic padding."
        )
    if vectors.shape[1] < final_dimension:
        vectors = np.pad(vectors, ((0, 0), (0, final_dimension - vectors.shape[1])))
    vectors = normalize(vectors.astype(np.float64, copy=False), norm="l2", axis=1)
    if not np.all(np.isfinite(vectors)):
        raise ValueError("Non-finite vectors detected")
    if np.any(np.linalg.norm(vectors, axis=1) < 1e-12):
        raise ValueError("At least one query/record is a zero vector after encoding")
    return vectors




def save_topk(
    database: np.ndarray,
    queries: np.ndarray,
    db_ids: list[str],
    query_ids: list[str],
    top_k: int,
    output_csv: Path,
) -> None:
    """L2 normalized: cosine similarity is simply query @ database.T."""
    scores = queries @ database.T
    rows = []
    for i, query_id in enumerate(query_ids):
        top = np.argsort(-scores[i], kind="stable")[:min(top_k, len(db_ids))]
        for rank, j in enumerate(top, start=1):
            rows.append({
                "query_id": query_id,
                "rank": rank,
                "record_id": db_ids[j],
                "cosine_similarity": round(float(scores[i, j]), 8),
            })
    pd.DataFrame(rows).to_csv(output_csv, index=False)


def run(
    csv_path: str | None = None,
    sample_size: int = 0,
    query_count: int = 20,
    pad_to: int = 32,
    top_k: int = 5,
    seed: int = 42,
    output_dir: Path = Path("data/processed/heart"),
) -> Path:
    raw = load_dataset(csv_path)
    print(f"Loaded Heart Disease: {len(raw):,} records")
    frame = prepare_columns(raw)
    if sample_size < 0:
        raise ValueError("--sample-size must be >= 0")
    if sample_size:
        frame = frame.sample(n=min(sample_size, len(frame)), random_state=seed)
    frame = frame.reset_index(drop=True)
    if not 1 <= query_count < len(frame):
        raise ValueError(f"--query-count must be 1..{len(frame)-1}")
    if top_k < 1:
        raise ValueError("--top-k must be at least 1")
    if pad_to < 0:
        raise ValueError("--pad-to must be >= 0")

    shuffled = np.random.default_rng(seed).permutation(len(frame))
    query_rows = frame.iloc[shuffled[:query_count]].reset_index(drop=True)
    db_rows = frame.iloc[shuffled[query_count:]].reset_index(drop=True)

    # Fit on DATABASE ONLY, then transform both database and queries identically.
    transformer = make_preprocessor()
    db_encoded = np.asarray(transformer.fit_transform(db_rows), dtype=np.float64)
    query_encoded = np.asarray(transformer.transform(query_rows), dtype=np.float64)
    encoded_dimension = db_encoded.shape[1]
    final_dimension = pad_to or (1 << (encoded_dimension - 1).bit_length())
    database = pad_and_normalize(db_encoded, final_dimension)
    queries = pad_and_normalize(query_encoded, final_dimension)

    db_ids = [f"H{i:05d}" for i in range(1, len(database) + 1)]
    query_ids = [f"Q{i:05d}" for i in range(1, len(queries) + 1)]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "database.npy", database)
    np.save(output_dir / "queries.npy", queries)
    # Headerless float64 CSV: one vector per row, suitable for C++ CSV loaders.
    np.savetxt(output_dir / "database_vectors.csv", database, delimiter=",", fmt="%.12g")
    np.savetxt(output_dir / "query_vectors.csv", queries, delimiter=",", fmt="%.12g")
    pd.DataFrame({"record_id": db_ids}).to_csv(output_dir / "database_ids.csv", index=False)
    pd.DataFrame({"query_id": query_ids}).to_csv(output_dir / "query_ids.csv", index=False)
    joblib.dump(transformer, output_dir / "preprocessor.joblib")
    save_topk(database, queries, db_ids, query_ids, top_k, output_dir / "plaintext_topk.csv")

    metadata = {
        "dataset": "heart",
        "source_uci_id": UCI_ID,
        "registered_records": len(database),
        "query_records": len(queries),
        "numeric_columns": NUMERIC_COLUMNS,
        "categorical_columns": CATEGORICAL_COLUMNS,
        "encoded_dimension": int(encoded_dimension),
        "padded_dimension": int(final_dimension),
        "zero_padding_adds_medical_information": False,
        "normalization": "L2 after imputation, scaling and one-hot encoding",
        "dtype": "float64",
        "csv_has_header": False,
        "outcome_label_used_as_feature": False,
        "top_k": top_k,
        "seed": seed,
        "clinical_validity": "NOT established; cosine similarity is mathematical similarity only",
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Registered: {database.shape}; queries: {queries.shape}")
    print(f"Encoded dimension: {encoded_dimension}; final dimension: {final_dimension}")
    print(f"Saved to: {output_dir.resolve()}")
    return output_dir


def cli() -> None:
    parser = argparse.ArgumentParser(description="CipherCare | Heart Disease preprocessing")
    parser.add_argument("--csv", dest="csv_path", type=str, default=None, help="Optional local CSV; otherwise download from UCI")
    parser.add_argument("--sample-size", type=int, default=0, help="0 = all records")
    parser.add_argument("--query-count", type=int, default=20)
    parser.add_argument("--pad-to", type=int, default=32, help="0 = next power of two automatically")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/heart"))
    args = parser.parse_args()
    try:
        run(**vars(args))
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    cli()
