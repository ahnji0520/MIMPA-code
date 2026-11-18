#!/usr/bin/env python3
"""Build deterministic 10-stream-task + 1-target episodes for E2E evaluation."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = ROOT / "data"
DATASETS = ("mmlu", "truthfulqa", "medmcqa", "mmlu_pro", "scalr")
LABELS4 = ("A", "B", "C", "D")


def option_labels(count: int) -> tuple[str, ...]:
    if not 2 <= count <= 10:
        raise ValueError(f"Unsupported option count: {count}")
    return tuple(chr(ord("A") + index) for index in range(count))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def stable_rng(*parts: Any) -> random.Random:
    raw = "::".join(map(str, parts)).encode("utf-8")
    return random.Random(int.from_bytes(hashlib.sha256(raw).digest()[:8], "big"))


def clean_value(value: Any) -> Any:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def mmlu_raw_rows() -> list[dict[str, Any]]:
    df = pd.read_parquet(DATA_ROOT / "mmlu/raw/mmlu_all_test.parquet")
    return [
        {
            "question": str(row["question"]),
            "subject": str(row["subject"]),
            "choices": list(row["choices"]),
            "answer": int(row["answer"]),
            "source_index": int(index),
            "source_id": f"mmlu_test_{index}",
            "source_split": "test",
        }
        for index, row in df.iterrows()
    ]


def truthfulqa_raw_rows() -> list[dict[str, Any]]:
    df = pd.read_csv(DATA_ROOT / "truthfulqa/raw/TruthfulQA.csv")
    rows = []
    for index, row in df.iterrows():
        record = {key: clean_value(value) for key, value in row.to_dict().items()}
        record.update(source_index=int(index), source_id=f"TruthfulQA/{index}")
        rows.append(record)
    return rows


def medmcqa_raw_rows() -> list[dict[str, Any]]:
    df = pd.read_parquet(DATA_ROOT / "medmcqa/raw/validation.parquet")
    rows = []
    for index, row in df.iterrows():
        record = {key: clean_value(value) for key, value in row.to_dict().items()}
        record.update(
            source_index=int(index),
            source_id=f"medmcqa_validation_{index}",
            eval_id=f"medmcqa_validation_{index}",
        )
        rows.append(record)
    return rows


def mmlu_pro_raw_rows() -> list[dict[str, Any]]:
    df = pd.read_parquet(DATA_ROOT / "mmlu_pro/raw/test.parquet")
    rows = []
    for source_index, row in df.iterrows():
        # Deliberately omit cot_content: it contains gold reasoning and must not
        # enter either task prompts or retrieval documents.
        rows.append({
            "question_id": int(row["question_id"]),
            "question": str(row["question"]),
            "options": list(row["options"]),
            "answer": str(row["answer"]),
            "answer_index": int(row["answer_index"]),
            "category": str(row["category"]),
            "src": str(row["src"]),
            "source_index": int(source_index),
            "source_id": f"mmlu_pro_test_{int(row['question_id']):05d}",
            "source_split": "test",
        })
    return rows


def scalr_raw_rows() -> list[dict[str, Any]]:
    df = pd.read_csv(DATA_ROOT / "scalr/raw/test.tsv", sep="\t")
    rows = []
    for source_index, row in df.iterrows():
        original_index = int(row["index"])
        rows.append({
            "index": original_index,
            "question": str(row["question"]),
            "options": [str(row[f"choice_{i}"]) for i in range(5)],
            "answer_index": int(row["answer"]),
            "category": "legal_reasoning",
            "source_index": int(source_index),
            "source_id": f"scalr_test_{original_index:05d}",
            "source_split": "test",
        })
    return rows


def normalize_mmlu(row: dict[str, Any]) -> dict[str, Any]:
    choices = {label: str(text) for label, text in zip(LABELS4, row["choices"])}
    return {
        "task_id": row.get("eval_id") or row["source_id"],
        "source_id": row["source_id"],
        "source_index": int(row["source_index"]),
        "subject": row["subject"],
        "question": str(row["question"]),
        "choices": choices,
        "labels": list(LABELS4),
        "gold_answer": LABELS4[int(row["answer"])],
    }


def normalize_truthfulqa(row: dict[str, Any], *, role: str, seed: int) -> dict[str, Any]:
    index = int(row["source_index"])
    pairs = [(str(row["Best Answer"]), True), (str(row["Best Incorrect Answer"]), False)]
    random.Random(seed * 1000 + index * 17 + (0 if role == "target" else 1)).shuffle(pairs)
    choices = {label: text for label, (text, _) in zip(("A", "B"), pairs)}
    gold = next(label for label, (_, is_gold) in zip(("A", "B"), pairs) if is_gold)
    return {
        "task_id": row.get("eval_id") or row["source_id"],
        "source_id": row["source_id"],
        "source_index": index,
        "subject": row["Category"],
        "type": row["Type"],
        "question": str(row["Question"]),
        "choices": choices,
        "labels": ["A", "B"],
        "gold_answer": gold,
        "best_answer": str(row["Best Answer"]),
        "best_incorrect_answer": str(row["Best Incorrect Answer"]),
    }


def normalize_medmcqa(row: dict[str, Any]) -> dict[str, Any]:
    answer = int(row["cop"])
    if answer not in range(4):
        raise ValueError(f"Invalid MedMCQA cop={answer} at {row['source_id']}")
    return {
        "task_id": row.get("eval_id") or row["source_id"],
        "source_id": row["source_id"],
        "source_index": int(row["source_index"]),
        "subject": row.get("subject_name") or "MedMCQA",
        "topic": row.get("topic_name"),
        "question": str(row["question"]),
        "choices": {
            "A": str(row["opa"]), "B": str(row["opb"]),
            "C": str(row["opc"]), "D": str(row["opd"]),
        },
        "labels": list(LABELS4),
        "gold_answer": LABELS4[answer],
    }


def normalize_variable_choice(row: dict[str, Any], *, subject_key: str) -> dict[str, Any]:
    labels = option_labels(len(row["options"]))
    answer_index = int(row["answer_index"])
    if answer_index not in range(len(labels)):
        raise ValueError(f"Invalid answer_index={answer_index} at {row['source_id']}")
    return {
        "task_id": row.get("eval_id") or row["source_id"],
        "source_id": row["source_id"],
        "source_index": int(row["source_index"]),
        "subject": row[subject_key],
        "question": str(row["question"]),
        "choices": {label: str(text) for label, text in zip(labels, row["options"])},
        "labels": list(labels),
        "gold_answer": labels[answer_index],
    }


def retrieval_text(task: dict[str, Any]) -> str:
    choices = "\n".join(f"{label}) {task['choices'][label]}" for label in task["labels"])
    return f"Question: {task['question']}\n{choices}"


def load_dataset(
    dataset: str,
    *,
    target_sample: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    targets = read_jsonl(DATA_ROOT / dataset / "sample" / target_sample / "targets.jsonl")
    if dataset == "mmlu":
        return targets, mmlu_raw_rows()
    if dataset == "truthfulqa":
        return targets, truthfulqa_raw_rows()
    if dataset == "medmcqa":
        return targets, medmcqa_raw_rows()
    if dataset == "mmlu_pro":
        return targets, mmlu_pro_raw_rows()
    if dataset == "scalr":
        return targets, scalr_raw_rows()
    raise ValueError(dataset)


def normalize(dataset: str, row: dict[str, Any], *, role: str, seed: int) -> dict[str, Any]:
    if dataset == "mmlu":
        return normalize_mmlu(row)
    if dataset == "truthfulqa":
        return normalize_truthfulqa(row, role=role, seed=seed)
    if dataset == "medmcqa":
        return normalize_medmcqa(row)
    if dataset == "mmlu_pro":
        return normalize_variable_choice(row, subject_key="category")
    if dataset == "scalr":
        return normalize_variable_choice(row, subject_key="category")
    raise ValueError(dataset)


def build(
    dataset: str,
    *,
    seed: int,
    stream_length: int,
    episode_count: int,
    target_sample: str,
    output_name: str,
) -> dict[str, Any]:
    target_rows, raw_rows = load_dataset(dataset, target_sample=target_sample)
    if len(target_rows) < episode_count:
        raise ValueError(f"{dataset}: only {len(target_rows)} targets, need {episode_count}")
    target_rows = target_rows[:episode_count]
    target_ids = {str(row["source_id"]) for row in target_rows}
    eligible = [row for row in raw_rows if str(row["source_id"]) not in target_ids]
    if len(eligible) < stream_length:
        raise ValueError(
            f"{dataset}: only {len(eligible)} non-Target rows, need {stream_length} per episode"
        )

    episodes: list[dict[str, Any]] = []
    flat: list[dict[str, Any]] = []
    for offset, target_row in enumerate(target_rows):
        episode_index = offset + 1
        target = normalize(dataset, target_row, role="target", seed=seed)
        # Retrieval is episode-local. Keep every ten-task stream internally
        # unique while allowing the same raw task to occur in different
        # episodes, which is required for smaller benchmarks such as TruthfulQA.
        block = stable_rng(seed, dataset, target["task_id"], "pre_target_tasks").sample(
            eligible,
            stream_length,
        )
        stream = [normalize(dataset, row, role="stream", seed=seed) for row in block]
        wrong = [label for label in target["labels"] if label != target["gold_answer"]]
        attack_answer = stable_rng(seed, dataset, target["task_id"], "attack_answer").choice(wrong)
        for position, task in enumerate(stream, 1):
            task["stream_position"] = position
            task["retrieval_text"] = retrieval_text(task)
            flat.append({"dataset": dataset, "episode_index": episode_index, "position": position, "task": task})
        target["retrieval_text"] = retrieval_text(target)
        episodes.append({
            "schema_version": 1,
            "dataset": dataset,
            "seed": seed,
            "episode_index": episode_index,
            "stream_length": stream_length,
            "pre_target_tasks": stream,
            "target": target,
            "attack_answer": attack_answer,
            "planting_positions": None,
        })

    out = DATA_ROOT / dataset / "sample" / output_name
    write_jsonl(out / "episodes.jsonl", episodes)
    write_jsonl(out / "stream_tasks.jsonl", flat)
    stream_ids = {r["task"]["source_id"] for r in flat}
    manifest = {
        "schema_version": 1,
        "dataset": dataset,
        "seed": seed,
        "episode_count": episode_count,
        "stream_length": stream_length,
        "target_count": len(episodes),
        "stream_task_count": len(flat),
        "unique_stream_task_count": len(stream_ids),
        "target_stream_overlap_count": len(target_ids & stream_ids),
        "target_source": str(DATA_ROOT / dataset / "sample" / target_sample / "targets.jsonl"),
        "raw_source": {
            "mmlu": str(DATA_ROOT / "mmlu/raw/mmlu_all_test.parquet"),
            "truthfulqa": str(DATA_ROOT / "truthfulqa/raw/TruthfulQA.csv"),
            "medmcqa": str(DATA_ROOT / "medmcqa/raw/validation.parquet"),
            "mmlu_pro": str(DATA_ROOT / "mmlu_pro/raw/test.parquet"),
            "scalr": str(DATA_ROOT / "scalr/raw/test.tsv"),
        }[dataset],
        "sampling": (
            f"For each Target, {stream_length} distinct pre-target tasks are "
            f"sampled deterministically after excluding all {episode_count} Targets; "
            "cross-episode reuse is allowed because retrieval is episode-local"
        ),
        "similarity_exposed_to_attacker": False,
        "files": {"episodes": "episodes.jsonl", "stream_tasks": "stream_tasks.jsonl"},
    }
    write_json(out / "manifest.json", manifest)
    return manifest


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--stream-length", type=int, default=10)
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument(
        "--target-sample",
        default=None,
        help="Target sample directory name; defaults to sample_<episodes>.",
    )
    p.add_argument(
        "--output-name",
        default=None,
        help="Output directory name; defaults to main_seed<seed>.",
    )
    args = p.parse_args()
    target_sample = args.target_sample or f"sample_{args.episodes}"
    output_name = args.output_name or f"main_seed{args.seed}"
    for dataset in args.datasets:
        print(json.dumps(build(
            dataset,
            seed=args.seed,
            stream_length=args.stream_length,
            episode_count=args.episodes,
            target_sample=target_sample,
            output_name=output_name,
        ), ensure_ascii=False))


if __name__ == "__main__":
    main()
