#!/usr/bin/env python3
"""Merge independently executed Main-experiment shards for one dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_main import summarize


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--expected-episodes", type=int, required=True)
    args = parser.parse_args()

    summaries: list[dict] = []
    manifests: list[dict] = []
    calls: list[str] = []
    for shard_id in range(args.num_shards):
        shard_root = args.dataset_root / "shards" / f"shard_{shard_id}"
        summaries.append(read_json(shard_root / "summary.json"))
        manifests.append(read_json(shard_root / "manifest.json"))
        audit_path = shard_root / "api_calls.jsonl"
        if audit_path.exists():
            calls.extend(audit_path.read_text(encoding="utf-8").splitlines())

    first = summaries[0]
    results = sorted(
        [result for summary in summaries for result in summary["results"]],
        key=lambda result: result["episode_index"],
    )
    indices = [result["episode_index"] for result in results]
    if len(indices) != len(set(indices)):
        raise RuntimeError(f"duplicate episode indices across shards: {indices}")
    if len(results) != args.expected_episodes:
        raise RuntimeError(
            f"expected {args.expected_episodes} merged episodes, found {len(results)}"
        )

    merged = summarize(first["dataset"], first["seed"], first["model"], results)
    write_json(args.dataset_root / "summary.json", merged)
    manifest = dict(manifests[0])
    manifest.update(
        {
            "episode_count": len(results),
            "total_episode_count": len(results),
            "shard_id": None,
            "num_shards": args.num_shards,
            "shard_episode_indices": indices,
            "shard_roots": [
                str(args.dataset_root / "shards" / f"shard_{shard_id}")
                for shard_id in range(args.num_shards)
            ],
        }
    )
    write_json(args.dataset_root / "manifest.json", manifest)
    (args.dataset_root / "api_calls.jsonl").write_text(
        "\n".join(calls) + ("\n" if calls else ""), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "dataset": first["dataset"],
                "episodes": len(results),
                "complete": merged["complete_episode_count"],
                "api_calls": len(calls),
                "output": str(args.dataset_root / "summary.json"),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
