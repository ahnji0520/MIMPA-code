#!/usr/bin/env python3
"""Run dataset Main experiments over independent OpenAI-compatible endpoints."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--base-urls", nargs="+", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--effort", default="off")
    parser.add_argument("--stream-name", default="stream_seed42_n10_s10")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--workers-per-job", type=int, default=1)
    parser.add_argument("--vllm-max-output-tokens", type=int, default=2048)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    num_shards = len(args.base_urls)
    if num_shards < 1:
        raise ValueError("at least one --base-urls value is required")
    args.output_root.mkdir(parents=True, exist_ok=True)

    jobs: list[tuple[str, int, subprocess.Popen, object, float]] = []
    commands: list[list[str]] = []
    for shard_id, base_url in enumerate(args.base_urls):
        for dataset in args.datasets:
            shard_root = args.output_root / dataset / "shards" / f"shard_{shard_id}"
            shard_root.mkdir(parents=True, exist_ok=True)
            command = [
                sys.executable,
                str(HERE / "run_main.py"),
                "--dataset", dataset,
                "--stream-name", args.stream_name,
                "--model", args.model,
                "--effort", args.effort,
                "--backend", "vllm",
                "--base-url", base_url,
                "--vllm-max-output-tokens", str(args.vllm_max_output_tokens),
                "--limit", str(args.limit),
                "--workers", str(args.workers_per_job),
                "--shard-id", str(shard_id),
                "--num-shards", str(num_shards),
                "--output-root", str(shard_root),
            ]
            commands.append(command)
            if args.dry_run:
                continue
            log_handle = (shard_root / "run.log").open("a", encoding="utf-8")
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
            jobs.append((dataset, shard_id, process, log_handle, time.monotonic()))

    if args.dry_run:
        print(json.dumps(commands, ensure_ascii=False, indent=2))
        return

    failures: list[dict] = []
    for dataset, shard_id, process, log_handle, started in jobs:
        returncode = process.wait()
        elapsed = time.monotonic() - started
        log_handle.close()
        status = {
            "dataset": dataset,
            "shard_id": shard_id,
            "returncode": returncode,
            "wall_seconds": round(elapsed, 3),
        }
        print(json.dumps(status, ensure_ascii=False), flush=True)
        if returncode:
            failures.append(status)

    if failures:
        raise SystemExit(f"{len(failures)} shard jobs failed: {failures}")

    for dataset in args.datasets:
        subprocess.run(
            [
                sys.executable,
                str(HERE / "merge_main_shards.py"),
                "--dataset-root", str(args.output_root / dataset),
                "--num-shards", str(num_shards),
                "--expected-episodes", str(args.limit),
            ],
            cwd=ROOT,
            check=True,
        )


if __name__ == "__main__":
    main()
