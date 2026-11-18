#!/usr/bin/env python3
"""Merge disjoint G-Memory episode shards into one dataset summary."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from run_gmemory import summarize, write_json


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--expected-episodes", type=int, required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()

    manifests: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    for shard_id in range(args.num_shards):
        shard_root = args.dataset_root / "shards" / f"shard_{shard_id}"
        manifests.append(read_json(shard_root / "manifest.json"))
        results.extend(read_json(shard_root / "summary.json")["results"])

    datasets = {row["dataset"] for row in manifests}
    models = {row["model"] for row in manifests}
    seeds = {int(row["seed"]) for row in manifests}
    if len(datasets) != 1 or len(models) != 1 or len(seeds) != 1:
        raise ValueError(
            f"inconsistent shard metadata: datasets={datasets}, "
            f"models={models}, seeds={seeds}"
        )

    indices = [int(row["episode_index"]) for row in results]
    if len(indices) != len(set(indices)):
        raise ValueError("duplicate episode indices across G-Memory shards")
    if len(results) != args.expected_episodes:
        raise ValueError(
            f"expected {args.expected_episodes} episodes, found {len(results)}"
        )
    incomplete = [row for row in results if not row.get("complete")]
    if incomplete and not args.allow_incomplete:
        details = [
            {"episode_index": row.get("episode_index"), "error": row.get("error")}
            for row in incomplete
        ]
        raise RuntimeError(f"{len(incomplete)} incomplete episode(s): {details}")

    results.sort(key=lambda row: int(row["episode_index"]))
    dataset = next(iter(datasets))
    model = next(iter(models))
    seed = next(iter(seeds))
    merged = summarize(dataset, seed, model, results)
    write_json(args.dataset_root / "summary.json", merged)
    write_json(
        args.dataset_root / "manifest.json",
        {
            **manifests[0],
            "episode_count": len(results),
            "total_episode_count": args.expected_episodes,
            "episode_workers": sum(
                int(row.get("episode_workers", 0)) for row in manifests
            ),
            "shard_id": None,
            "num_shards": args.num_shards,
            "shard_episode_indices": None,
            "merged_from": [
                str(args.dataset_root / "shards" / f"shard_{i}")
                for i in range(args.num_shards)
            ],
        },
    )
    print(
        json.dumps(
            {
                "dataset": dataset,
                "model": model,
                "episodes": len(results),
                "complete": len(results) - len(incomplete),
                "output": str(args.dataset_root / "summary.json"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
