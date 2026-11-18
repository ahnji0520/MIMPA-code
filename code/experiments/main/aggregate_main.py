#!/usr/bin/env python3
"""Aggregate dataset-level Main results from one or more memory/model runs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

DATASETS = ("mmlu_pro", "truthfulqa", "medmcqa")
CONDITIONS = (
    "clean",
    "mac_attack",
    "autotransform",
    "mimpa_wo_poisoned_record",
    "mimpa_wo_triggering",
    "mimpa",
)


def parse_source(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--source must be LABEL=/path/to/model/root")
    label, path = value.split("=", 1)
    return label.strip(), Path(path).expanduser()


def read_summary(root: Path, dataset: str) -> dict[str, Any]:
    path = root / dataset / "summary.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source", action="append", type=parse_source, required=True,
        help="Repeatable LABEL=/path/to/model/root; root contains dataset folders.",
    )
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    source_metadata: dict[str, Any] = {}
    for label, root in args.source:
        source_metadata[label] = str(root.resolve())
        for dataset in args.datasets:
            summary = read_summary(root, dataset)
            backend = (
                summary.get("memory_backend")
                or next(
                    (
                        row.get("memory_backend")
                        for row in summary.get("results", [])
                        if row.get("memory_backend")
                    ),
                    label,
                )
            )
            for condition in CONDITIONS:
                metric = summary["conditions"][condition]
                rows.append({
                    "source": label,
                    "memory_backend": backend,
                    "model": summary["model"],
                    "dataset": dataset,
                    "condition": condition,
                    "accuracy_numerator": metric["accuracy_numerator"],
                    "accuracy_denominator": metric["accuracy_denominator"],
                    "accuracy": metric["accuracy"],
                    "asr_numerator": metric["asr_numerator"],
                    "asr_denominator": metric[
                        "asr_denominator_fixed_clean_correct"
                    ],
                    "asr": metric["asr"],
                })
    result = {
        "sources": source_metadata,
        "datasets": args.datasets,
        "condition_order": list(CONDITIONS),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
