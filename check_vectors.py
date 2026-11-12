### 결과 검증
from pathlib import Path
import argparse

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/heart"))
    args = parser.parse_args()
    folder = args.data_dir
    database = np.load(folder / "database.npy")
    queries = np.load(folder / "queries.npy")
    ids = pd.read_csv(folder / "database_ids.csv")["record_id"].tolist()

    assert database.ndim == queries.ndim == 2
    assert database.shape[1] == queries.shape[1]
    assert np.allclose(np.linalg.norm(database, axis=1), 1.0, atol=1e-7)
    assert np.allclose(np.linalg.norm(queries, axis=1), 1.0, atol=1e-7)

    scores = database @ queries[0]
    top = np.argsort(-scores, kind="stable")[:5]
    print("Dataset: UCI Heart Disease")
    print(f"Database: {database.shape[0]} records; queries: {queries.shape[0]}")
    print(f"Vector dimension: {database.shape[1]}")
    print("Query #1 | Top-5:")
    for rank, index in enumerate(top, 1):
        print(f"  {rank}. {ids[index]}  cosine={scores[index]:.6f}")
    print("PASSED: vector shapes and L2 norms verified.")


if __name__ == "__main__":
    main()
