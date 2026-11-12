"""Transform a new Heart Disease query with the existing fitted transformer.

Only load joblib models saved locally and trusted by your team.
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from preprocess import pad_and_normalize, prepare_columns


def main() -> None:
    parser = argparse.ArgumentParser(description="Transform a new Heart Disease query")
    parser.add_argument("--csv", type=Path, required=True, help="CSV with 13 Heart Disease input columns")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/heart"))
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    metadata = json.loads((args.data_dir / "metadata.json").read_text(encoding="utf-8"))
    transformer = joblib.load(args.data_dir / "preprocessor.joblib")
    raw = pd.read_csv(args.csv)
    medical_features = prepare_columns(raw)
    encoded = np.asarray(transformer.transform(medical_features), dtype=np.float64)
    vectors = pad_and_normalize(encoded, metadata["padded_dimension"])

    destination = args.out or args.data_dir / "new_query_vectors.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(destination, vectors, delimiter=",", fmt="%.12g")
    print(f"Saved {len(vectors)} Heart Disease query vectors -> {destination}")
    print("PLAINTEXT: encrypt locally before sending to the server.")


if __name__ == "__main__":
    main()
