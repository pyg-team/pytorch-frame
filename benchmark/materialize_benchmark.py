"""Benchmark materialize() performance across DataFrame backends.

Usage:
    python benchmark/materialize_benchmark.py \\
        --backend pandas \\
        --num_rows 100000 \\
        --stypes numerical categorical multicategorical timestamp

    python benchmark/materialize_benchmark.py \\
        --backend cudf \\
        --num_rows 1000000

    # Compare both backends on the same workload:
    python benchmark/materialize_benchmark.py --compare
"""
import argparse
import random
import string
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

import torch_frame
from torch_frame import stype
from torch_frame.data import Dataset

STYPE_CHOICES = [
    "numerical",
    "categorical",
    "multicategorical",
    "timestamp",
    "sequence_numerical",
    "embedding",
]

TIME_FORMATS = ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%Y/%m/%d"]


def _random_timestamp(start: datetime, end: datetime, fmt: str) -> str:
    ts = start + timedelta(
        seconds=random.randint(0, int((end - start).total_seconds())))
    return ts.strftime(fmt)


def generate_dataframe(
    num_rows: int,
    stypes: list[str],
) -> tuple[pd.DataFrame, dict[str, stype]]:
    """Generate a synthetic pandas DataFrame for benchmarking."""
    df_dict: dict[str, list | np.ndarray] = {}
    col_to_stype: dict[str, stype] = {}
    col_to_sep: dict[str, str] = {}

    # Always include a numerical target
    df_dict["target"] = np.random.randn(num_rows)
    col_to_stype["target"] = stype.numerical

    if "numerical" in stypes:
        for i in range(5):
            col = f"num_{i}"
            df_dict[col] = np.random.randn(num_rows)
            col_to_stype[col] = stype.numerical

    if "categorical" in stypes:
        for i in range(5):
            col = f"cat_{i}"
            df_dict[col] = np.random.randint(0, 20, size=num_rows)
            col_to_stype[col] = stype.categorical

    if "multicategorical" in stypes:
        vocab = list(string.ascii_lowercase[:10])
        for i in range(3):
            col = f"multicat_{i}"
            arr = []
            for _ in range(num_rows):
                k = random.randint(1, 4)
                arr.append(",".join(random.sample(vocab, k)))
            df_dict[col] = arr
            col_to_stype[col] = stype.multicategorical
            col_to_sep[col] = ","

    if "timestamp" in stypes:
        start_date = datetime(2000, 1, 1)
        end_date = datetime(2024, 1, 1)
        for i, fmt in enumerate(TIME_FORMATS):
            col = f"timestamp_{i}"
            df_dict[col] = [
                _random_timestamp(start_date, end_date, fmt)
                for _ in range(num_rows)
            ]
            col_to_stype[col] = stype.timestamp

    if "sequence_numerical" in stypes:
        for i in range(3):
            col = f"seq_num_{i}"
            arr = []
            for _ in range(num_rows):
                length = random.randint(1, 8)
                arr.append([random.random() for _ in range(length)])
            df_dict[col] = arr
            col_to_stype[col] = stype.sequence_numerical

    if "embedding" in stypes:
        for i in range(2):
            col = f"emb_{i}"
            emb_dim = 8
            df_dict[col] = [
                [random.random() for _ in range(emb_dim)]
                for _ in range(num_rows)
            ]
            col_to_stype[col] = stype.embedding

    return pd.DataFrame(df_dict), col_to_stype, col_to_sep


def benchmark_materialize(
    df,
    col_to_stype,
    col_to_sep,
    col_to_time_format,
    n_runs: int = 3,
) -> list[float]:
    """Time materialize() over n_runs, return list of wall times."""
    times = []
    for _ in range(n_runs):
        dataset = Dataset(
            df.copy(),
            col_to_stype,
            target_col="target",
            col_to_sep=col_to_sep,
            col_to_time_format=col_to_time_format,
        )
        t0 = time.perf_counter()
        dataset.materialize()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return times


def run_benchmark(args):
    print(f"Generating DataFrame: {args.num_rows:,} rows, "
          f"stypes={args.stypes}")
    df_pandas, col_to_stype, col_to_sep = generate_dataframe(
        args.num_rows, args.stypes)

    col_to_time_format = {}
    for col, st in col_to_stype.items():
        if st == stype.timestamp:
            idx = int(col.split("_")[-1])
            col_to_time_format[col] = TIME_FORMATS[idx]

    print(f"DataFrame shape: {df_pandas.shape}")
    print(f"Columns: { {s: sum(1 for v in col_to_stype.values() if v == s) for s in set(col_to_stype.values())} }")
    print()

    backends = (
        ["pandas", "cudf"] if args.compare
        else [args.backend]
    )

    results = {}
    for backend in backends:
        if backend == "cudf":
            try:
                import cudf
                df = cudf.DataFrame.from_pandas(df_pandas)
            except ImportError:
                print("cudf not installed, skipping cudf backend")
                continue
        else:
            df = df_pandas

        print(f"--- {backend} ---")
        times = benchmark_materialize(
            df, col_to_stype, col_to_sep, col_to_time_format,
            n_runs=args.n_runs,
        )
        mean_t = np.mean(times)
        std_t = np.std(times)
        print(f"  materialize() times: "
              f"{[f'{t:.3f}s' for t in times]}")
        print(f"  mean: {mean_t:.3f}s +/- {std_t:.3f}s")
        print()
        results[backend] = (mean_t, std_t)

    if len(results) == 2:
        pd_mean = results["pandas"][0]
        cu_mean = results["cudf"][0]
        speedup = pd_mean / cu_mean
        print(f"Speedup (pandas / cudf): {speedup:.2f}x")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark materialize() across DataFrame backends")
    parser.add_argument(
        "--backend", choices=["pandas", "cudf"], default="pandas",
        help="DataFrame backend (default: pandas)")
    parser.add_argument(
        "--compare", action="store_true",
        help="Run both pandas and cudf and compare")
    parser.add_argument(
        "--num_rows", type=int, default=100_000,
        help="Number of rows (default: 100000)")
    parser.add_argument(
        "--stypes", nargs="+", choices=STYPE_CHOICES,
        default=["numerical", "categorical", "multicategorical", "timestamp"],
        help="Column types to include")
    parser.add_argument(
        "--n_runs", type=int, default=3,
        help="Number of runs per backend (default: 3)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    run_benchmark(args)


if __name__ == "__main__":
    main()
