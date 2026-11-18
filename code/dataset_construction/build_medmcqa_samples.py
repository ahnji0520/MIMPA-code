#!/usr/bin/env python3
"""Build deterministic MedMCQA Target samples from the validation split."""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "data" / "medmcqa"


def clean(value: Any) -> Any:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--output-name", default="sample_100")
    args = parser.parse_args()

    source = DATA_ROOT / "raw" / "validation.parquet"
    frame = pd.read_parquet(source)
    if args.count > len(frame):
        raise ValueError(f"Requested {args.count} rows from a pool of {len(frame)}")
    indices = random.Random(args.seed).sample(range(len(frame)), args.count)
    rows: list[dict[str, Any]] = []
    for rank, source_index in enumerate(indices, 1):
        row = {key: clean(value) for key, value in frame.iloc[source_index].to_dict().items()}
        row.update({
            "source_index": source_index,
            "source_id": f"medmcqa_validation_{source_index}",
            "eval_id": f"medmcqa_eval_{rank:03d}",
            "sample_100_rank": rank,
        })
        rows.append(row)

    output = DATA_ROOT / "sample" / args.output_name
    write_jsonl(output / "targets.jsonl", rows)
    write_json(output / "summary.json", {
        "dataset": "openlifescienceai/medmcqa",
        "source": str(source),
        "source_split": "validation",
        "source_row_count": len(frame),
        "seed": args.seed,
        "sample_count": args.count,
        "sampling": f"random.Random({args.seed}).sample(range({len(frame)}), {args.count})",
        "source_indices": indices,
        "input_fields": ["question", "opa", "opb", "opc", "opd"],
        "label_field": "cop",
        "excluded_from_task_input": ["exp"],
    })
    print(json.dumps({"output": str(output), "count": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
