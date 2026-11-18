#!/usr/bin/env python3
"""Validate main episode and similarity-cache invariants without API calls."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
DATASETS = ("mmlu", "truthfulqa", "medmcqa", "mmlu_pro", "scalr")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def validate(
    dataset: str,
    seed: int,
    stream_name: str,
    expected_episodes: int,
    require_similarity: bool,
) -> dict[str, Any]:
    root = ROOT / "data" / dataset / "sample" / stream_name
    episodes = read_jsonl(root / "episodes.jsonl")
    assert len(episodes) == expected_episodes
    targets = [episode["target"]["source_id"] for episode in episodes]
    streams = [task["source_id"] for episode in episodes for task in episode["pre_target_tasks"]]
    assert len(targets) == len(set(targets)) == expected_episodes
    assert len(streams) == expected_episodes * 10
    assert not (set(targets) & set(streams))
    for episode in episodes:
        assert len(episode["pre_target_tasks"]) == 10
        assert len({task["source_id"] for task in episode["pre_target_tasks"]}) == 10
        assert [task["stream_position"] for task in episode["pre_target_tasks"]] == list(range(1, 11))
        assert episode["attack_answer"] in episode["target"]["labels"]
        assert episode["attack_answer"] != episode["target"]["gold_answer"]
        for task in episode["pre_target_tasks"] + [episode["target"]]:
            assert task["gold_answer"] in task["labels"]
            assert task["retrieval_text"].startswith("Question:")

    similarity_rows = 0
    if require_similarity:
        rows = read_jsonl(root / "similarities_qwen3_embedding_8b.jsonl")
        similarity_rows = len(rows)
        assert len(rows) == expected_episodes
        for row, episode in zip(rows, episodes):
            assert row["episode_index"] == episode["episode_index"]
            assert len(row["task_ids"]) == 11
            assert len(row["query_document_cosine_matrix"]) == 11
            assert all(len(vector) == 11 for vector in row["query_document_cosine_matrix"])
            assert len(row["rankings"]) == 11
            assert len(row["rankings"][-1]["ranked_predecessors"]) == 10
    return {
        "dataset": dataset,
        "episodes": len(episodes),
        "targets": len(targets),
        "unique_stream_tasks": len(set(streams)),
        "overlap": 0,
        "similarity_rows": similarity_rows,
        "valid": True,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--stream-name", default="stream_seed42_n100_s10")
    p.add_argument("--expected-episodes", type=int, default=100)
    p.add_argument("--require-similarity", action="store_true")
    args = p.parse_args()
    for dataset in args.datasets:
        print(json.dumps(validate(
            dataset,
            args.seed,
            args.stream_name,
            args.expected_episodes,
            args.require_similarity,
        ), ensure_ascii=False))


if __name__ == "__main__":
    main()
