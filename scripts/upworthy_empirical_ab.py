import os

# Must be set before importing numpy/scipy.
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import hashlib
import json
import logging
import pickle
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path

_SCRIPT_DIR = Path(__file__).parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
_SRC_PATH = _PROJECT_ROOT / "src"
if str(_SRC_PATH) not in sys.path:
    sys.path.insert(0, str(_SRC_PATH))

import numpy as np
import pandas as pd
import scipy
from joblib import Parallel, delayed

from winners_curse.empirical_ab import (
    ESTIMATOR_KEYS,
    estimate_aggregate_test,
    filter_eligible_tests,
    prepare_arm_table,
    stack_draws,
    summarize_estimates,
)


DEFAULT_ESTIMATORS = [
    "standard_bootstrap",
    "moon_bootstrap",
    "sample_splitting",
    "eb_normal",
    "hybrid_si",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run aggregate-count winner's curse estimators on Upworthy A/B tests."
    )
    parser.add_argument(
        "--input",
        default="dataset/upworthy/derived/deployed-arm-audit.csv",
        help="Arm-level Upworthy CSV.",
    )
    parser.add_argument(
        "--output",
        default="results/empirical_study/upworthy",
        help="Base output directory for timestamped runs.",
    )
    parser.add_argument("--n-bootstraps", type=int, default=1000)
    parser.add_argument("--n-sample-splits", type=int, default=1000)
    parser.add_argument("--moon-power", type=float, default=0.6)
    parser.add_argument("--estimation-split", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20260628)
    parser.add_argument("--max-jobs", type=int, default=24)
    parser.add_argument(
        "--estimators",
        nargs="+",
        default=DEFAULT_ESTIMATORS,
        choices=ESTIMATOR_KEYS,
        help="Estimator keys to run.",
    )
    parser.add_argument(
        "--limit-tests",
        type=int,
        default=None,
        help="Run only the first N eligible tests for smoke testing.",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Existing run directory with checkpoints to resume.",
    )
    parser.add_argument(
        "--checkpoint-size",
        type=int,
        default=500,
        help="Number of tests per checkpoint chunk.",
    )
    parser.add_argument("--min-arms", type=int, default=2)
    parser.add_argument("--min-impressions", type=int, default=1000)
    parser.add_argument(
        "--hybrid-n-simulations",
        type=int,
        default=10000,
        help="Monte Carlo draws for the hybrid selective-inference critical value.",
    )
    return parser.parse_args()


def setup_logging(output_dir: Path) -> logging.Logger:
    logger = logging.getLogger("upworthy_empirical_ab")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    log_file = output_dir / "experiment.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def resolve_path(path: str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = _PROJECT_ROOT / candidate
    return candidate.resolve()


def create_or_resume_output_dir(args: argparse.Namespace) -> Path:
    if args.resume:
        output_dir = resolve_path(args.resume)
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    base_dir = resolve_path(args.output)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = base_dir / f"upworthy_empirical_ab_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_git_command(args: list[str]) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=_PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def write_json(path: Path, payload: dict) -> None:
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=json_default)


def make_manifest(
    args: argparse.Namespace,
    input_path: Path,
    output_dir: Path,
    eligibility: dict,
    n_tests_run: int,
) -> dict:
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "command": sys.argv,
        "project_root": str(_PROJECT_ROOT),
        "input_path": str(input_path),
        "input_sha256": sha256_file(input_path),
        "output_dir": str(output_dir),
        "git_commit": run_git_command(["rev-parse", "HEAD"]),
        "git_status_short": run_git_command(["status", "--short"]),
        "python": platform.python_version(),
        "packages": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
        },
        "parameters": {
            "estimators": list(args.estimators),
            "n_bootstraps": args.n_bootstraps,
            "n_sample_splits": args.n_sample_splits,
            "moon_power": args.moon_power,
            "estimation_split": args.estimation_split,
            "seed": args.seed,
            "max_jobs": args.max_jobs,
            "limit_tests": args.limit_tests,
            "checkpoint_size": args.checkpoint_size,
            "hybrid_n_simulations": args.hybrid_n_simulations,
        },
        "eligibility": eligibility,
        "n_tests_run": n_tests_run,
    }


def checkpoint_path(checkpoint_dir: Path, start: int, end: int) -> Path:
    return checkpoint_dir / f"chunk_{start:06d}_{end:06d}.pkl"


def checkpoint_signature(args: argparse.Namespace) -> dict:
    return {
        "estimators": list(args.estimators),
        "n_bootstraps": args.n_bootstraps,
        "n_sample_splits": args.n_sample_splits,
        "moon_power": args.moon_power,
        "estimation_split": args.estimation_split,
        "seed": args.seed,
        "hybrid_n_simulations": args.hybrid_n_simulations,
    }


def save_checkpoint(path: Path, payload: dict) -> None:
    tmp_path = path.with_suffix(".tmp")
    with tmp_path.open("wb") as handle:
        pickle.dump(payload, handle)
    tmp_path.replace(path)


def load_checkpoint(path: Path) -> dict:
    with path.open("rb") as handle:
        return pickle.load(handle)


def run_chunk(
    chunk_items: list[tuple[int, str, pd.DataFrame]],
    checkpoint_dir: Path,
    args: argparse.Namespace,
    logger: logging.Logger,
) -> dict:
    start = chunk_items[0][0]
    end = chunk_items[-1][0] + 1
    ckpt_path = checkpoint_path(checkpoint_dir, start, end)
    if args.resume and ckpt_path.exists():
        logger.info("Loading checkpoint %s", ckpt_path)
        payload = load_checkpoint(ckpt_path)
        if payload.get("signature") != checkpoint_signature(args):
            raise ValueError(
                f"Checkpoint parameter mismatch for {ckpt_path}. "
                "Start a new run directory or rerun with matching estimator parameters."
            )
        return payload

    logger.info("Running tests %s-%s (%s tests)", start, end - 1, len(chunk_items))

    def run_one(item: tuple[int, str, pd.DataFrame]):
        test_index, _test_id, group = item
        return estimate_aggregate_test(
            group=group,
            test_index=test_index,
            seed=args.seed,
            estimators=args.estimators,
            n_bootstraps=args.n_bootstraps,
            n_sample_splits=args.n_sample_splits,
            moon_power=args.moon_power,
            estimation_split=args.estimation_split,
            hybrid_n_simulations=args.hybrid_n_simulations,
        )

    if args.max_jobs == 1:
        results = [run_one(item) for item in chunk_items]
    else:
        results = Parallel(n_jobs=args.max_jobs, verbose=0)(
            delayed(run_one)(item) for item in chunk_items
        )

    records = [record for record, _draws in results]
    draw_dicts = [draws for _record, draws in results]
    payload = {
        "start": start,
        "end": end,
        "signature": checkpoint_signature(args),
        "test_ids": [test_id for _idx, test_id, _group in chunk_items],
        "records": records,
        "draws": stack_draws(draw_dicts),
    }
    save_checkpoint(ckpt_path, payload)
    logger.info("Saved checkpoint %s", ckpt_path)
    return payload


def chunked(items: list, size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def save_outputs(
    output_dir: Path,
    eligible_arms: pd.DataFrame,
    per_test: pd.DataFrame,
    draw_chunks: list[dict[str, np.ndarray]],
    estimators: list[str],
) -> tuple[pd.DataFrame, dict]:
    arm_path = output_dir / "arm_level_input.csv.gz"
    per_test_path = output_dir / "per_test_estimates.csv.gz"
    draws_path = output_dir / "estimator_draws.npz"
    summary_table_path = output_dir / "summary_tables.csv"
    summary_json_path = output_dir / "summary.json"

    eligible_arms.to_csv(arm_path, index=False, compression="gzip")
    per_test.to_csv(per_test_path, index=False, compression="gzip")

    draw_arrays: dict[str, np.ndarray] = {}
    if draw_chunks:
        for key in sorted({key for chunk in draw_chunks for key in chunk}):
            draw_arrays[key] = np.concatenate(
                [chunk[key] for chunk in draw_chunks if key in chunk],
                axis=0,
            )
    draw_arrays["test_ids"] = per_test["clickability_test_id"].to_numpy(dtype=str)
    np.savez_compressed(draws_path, **draw_arrays)

    summary_table, summary_json = summarize_estimates(per_test, estimators)
    summary_table.to_csv(summary_table_path, index=False)
    write_json(summary_json_path, summary_json)
    return summary_table, {
        "arm_level_input": str(arm_path),
        "per_test_estimates": str(per_test_path),
        "estimator_draws": str(draws_path),
        "summary_tables": str(summary_table_path),
        "summary_json": str(summary_json_path),
    }


def main() -> None:
    args = parse_args()
    if args.n_bootstraps <= 0:
        raise SystemExit("--n-bootstraps must be positive.")
    if args.n_sample_splits <= 0:
        raise SystemExit("--n-sample-splits must be positive.")
    if args.checkpoint_size <= 0:
        raise SystemExit("--checkpoint-size must be positive.")
    if args.max_jobs <= 0:
        raise SystemExit("--max-jobs must be positive.")

    input_path = resolve_path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    output_dir = create_or_resume_output_dir(args)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir)

    logger.info("Reading %s", input_path)
    raw = pd.read_csv(input_path)
    arms = prepare_arm_table(raw)
    eligible_arms, eligibility = filter_eligible_tests(
        arms,
        min_arms=args.min_arms,
        min_impressions=args.min_impressions,
        limit_tests=args.limit_tests,
    )
    test_items = [
        (idx, test_id, group.copy())
        for idx, (test_id, group) in enumerate(
            eligible_arms.groupby("clickability_test_id", sort=True)
        )
    ]

    manifest = make_manifest(
        args=args,
        input_path=input_path,
        output_dir=output_dir,
        eligibility=eligibility.to_dict(),
        n_tests_run=len(test_items),
    )
    write_json(output_dir / "run_manifest.json", manifest)

    logger.info(
        "Eligible tests: %s of %s; running %s tests",
        eligibility.eligible_tests,
        eligibility.total_tests,
        len(test_items),
    )
    logger.info("Estimators: %s", ", ".join(args.estimators))

    payloads = []
    for chunk_items in chunked(test_items, args.checkpoint_size):
        payloads.append(run_chunk(chunk_items, checkpoint_dir, args, logger))

    records = [record for payload in payloads for record in payload["records"]]
    per_test = pd.DataFrame(records)
    per_test = per_test.sort_values("clickability_test_id", kind="mergesort").reset_index(
        drop=True
    )
    draw_chunks = [payload["draws"] for payload in payloads]
    _summary_table, output_files = save_outputs(
        output_dir=output_dir,
        eligible_arms=eligible_arms,
        per_test=per_test,
        draw_chunks=draw_chunks,
        estimators=list(args.estimators),
    )

    manifest["completed_at"] = datetime.now().isoformat(timespec="seconds")
    manifest["output_files"] = output_files
    write_json(output_dir / "run_manifest.json", manifest)
    logger.info("Results saved to %s", output_dir)


if __name__ == "__main__":
    main()
