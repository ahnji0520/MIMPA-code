"""Read-only similarity cache and successful-memory Top-k retrieval."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class SimilarityCache:
    def __init__(self, path: Path):
        self.path = path
        self.episodes = {int(row["episode_index"]): row for row in read_jsonl(path)}

    def retrieve(
        self,
        *,
        episode_index: int,
        query_task_id: str,
        successful_logs: dict[str, dict[str, Any]],
        top_k: int = 3,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        episode = self.episodes[episode_index]
        ranking = next(
            item for item in episode["rankings"] if item["query_task_id"] == query_task_id
        )["ranked_predecessors"]
        eligible = [row for row in ranking if row["task_id"] in successful_logs]
        selected = eligible[:top_k]
        return [successful_logs[row["task_id"]] for row in selected], selected


def serialize_retrieved(adapter: Any, logs: list[dict[str, Any]]) -> str:
    return "\n\n".join(adapter.serialize_memory(log) for log in logs)
