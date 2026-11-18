#!/usr/bin/env python3
"""Plan or execute one model across datasets and both memory backends.

Execution is opt-in. Without --execute this prints the exact commands and does
not call any model API.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
DATASETS = ("mmlu_pro", "truthfulqa", "medmcqa")
MEMORY_BACKENDS = ("shared", "gmemory")


def model_slug(model: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in model).strip("_")


def shared_model_root(args: argparse.Namespace) -> Path:
    run_id = f"seed42_n{args.sample_size}_s10_k3_r2"
    return (
        ROOT / "data/results/main_v2/shared" / run_id / model_slug(args.model)
    )


def build_command(args: argparse.Namespace, memory_backend: str, dataset: str) -> list[str]:
    stream = f"stream_seed42_n{args.sample_size}_s10"
    if memory_backend == "shared":
        runner = ROOT / "code/experiments/main/run_main.py"
    else:
        runner = ROOT / "code/experiments/gmemory/run_gmemory.py"
    command = [
        sys.executable, str(runner),
        "--dataset", dataset,
        "--stream-name", stream,
        "--limit", str(args.sample_size),
        "--model", args.model,
        "--effort", args.effort,
        "--backend", args.backend,
        "--workers", str(args.workers),
        "--api-key-env", args.api_key_env,
    ]
    if args.backend == "vllm":
        command += [
            "--base-url", args.base_url,
            "--vllm-max-output-tokens", str(args.vllm_max_output_tokens),
        ]
    reuse = (
        args.shared_reuse_clean_root
        if memory_backend == "shared"
        else args.gmemory_reuse_clean_root
    )
    if reuse is not None:
        command += ["--reuse-clean-root", str(reuse)]
    stream_reuse = (
        args.shared_reuse_clean_stream_root
        if memory_backend == "shared"
        else args.gmemory_reuse_clean_stream_root
    )
    if stream_reuse is not None:
        command += ["--reuse-clean-stream-root", str(stream_reuse)]
    if memory_backend == "gmemory":
        paired_payload_root = args.paired_payload_root
        if paired_payload_root is None and "shared" in args.memory_backends:
            paired_payload_root = shared_model_root(args)
        if paired_payload_root is not None:
            command += ["--payload-root", str(paired_payload_root)]
        command += [
            "--attack-opportunities", "3",
            "--retrieval-top-k", str(args.gmemory_retrieval_top_k),
            "--insight-top-k", str(args.gmemory_insight_top_k),
            "--retrieval-threshold", "0.0",
            "--hop", "1",
            "--start-insights-threshold", "5",
            "--rounds-per-insights", "5",
            "--insights-point-num", "5",
            "--manager-prompt-mode", "qa_adapted",
            "--embedding-device", args.embedding_device,
        ]
    return command


def run(command: list[str]) -> dict[str, Any]:
    process = subprocess.run(command, cwd=ROOT, text=True)
    return {"command": command, "returncode": process.returncode}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--effort", default="low")
    parser.add_argument("--backend", choices=("openai", "vllm"), default="openai")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--vllm-max-output-tokens", type=int, default=2048)
    parser.add_argument("--sample-size", type=int, choices=(10, 100), required=True)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument(
        "--memory-backends", nargs="+", choices=MEMORY_BACKENDS,
        default=list(MEMORY_BACKENDS),
    )
    parser.add_argument("--workers", type=int, default=30)
    parser.add_argument("--parallel-jobs", type=int, default=1)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--shared-reuse-clean-root", type=Path)
    parser.add_argument("--gmemory-reuse-clean-root", type=Path)
    parser.add_argument("--shared-reuse-clean-stream-root", type=Path)
    parser.add_argument("--gmemory-reuse-clean-stream-root", type=Path)
    parser.add_argument("--paired-payload-root", type=Path)
    parser.add_argument("--gmemory-retrieval-top-k", type=int, default=1)
    parser.add_argument("--gmemory-insight-top-k", type=int, default=3)
    parser.add_argument("--embedding-device", default="cpu")
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    commands = [
        build_command(args, memory_backend, dataset)
        for memory_backend in args.memory_backends
        for dataset in args.datasets
    ]
    plan = {
        "execute": args.execute,
        "model": args.model,
        "sample_size": args.sample_size,
        "datasets": args.datasets,
        "memory_backends": args.memory_backends,
        "autotransform_profile_policy": (
            "load the single frozen gpt-3.5-turbo transformed profile; never regenerate"
        ),
        "gmemory_config": {
            "retrieval_top_k": args.gmemory_retrieval_top_k,
            "insight_top_k": args.gmemory_insight_top_k,
            "retrieval_threshold": 0.0,
            "hop": 1,
            "start_insights_threshold": 5,
            "rounds_per_insights": 5,
            "insights_point_num": 5,
            "manager_prompt_mode": "qa_adapted",
        },
        "commands": commands,
    }
    print(json.dumps(plan, ensure_ascii=False, indent=2))
    if not args.execute:
        return
    rows: list[dict[str, Any]] = []
    # Shared runs produce the frozen per-Target payload map consumed by the
    # paired G-Memory runs. Execute the two phases in that order.
    for phase in ("shared", "gmemory"):
        phase_commands = [
            command
            for command in commands
            if (
                ("gmemory/run_gmemory.py" in command[1]) == (phase == "gmemory")
            )
        ]
        if not phase_commands:
            continue
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(args.parallel_jobs, len(phase_commands))
        ) as pool:
            phase_rows = list(pool.map(run, phase_commands))
        rows.extend(phase_rows)
        if any(row["returncode"] for row in phase_rows):
            break
    failures = [row for row in rows if row["returncode"]]
    if failures:
        raise SystemExit(
            "failed commands:\n" + "\n".join(" ".join(row["command"]) for row in failures)
        )


if __name__ == "__main__":
    main()
