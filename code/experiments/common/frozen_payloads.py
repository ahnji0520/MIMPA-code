"""Load already generated per-Target payloads for paired backend comparisons."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def dataset_root(root: Path, dataset: str) -> Path:
    nested = root / dataset
    if nested.is_dir():
        return nested
    if root.name == dataset and root.is_dir():
        return root
    return nested


def load_payload_map(
    root: Path | None, *, dataset: str,
) -> tuple[dict[str, str] | None, dict[str, Any] | None]:
    if root is None:
        return None, None
    base = dataset_root(root, dataset)
    episode_dirs = sorted(base.glob("episode_*"))
    if not episode_dirs:
        episode_dirs = sorted((base / "shards").glob("shard_*/episode_*"))
    values: dict[str, str] = {}
    files: dict[str, str] = {}
    for episode_dir in episode_dirs:
        summary_path = episode_dir / "summary.json"
        payload_path = episode_dir / "payload.txt"
        if not summary_path.exists() or not payload_path.exists():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        target_id = str(summary.get("target_id", ""))
        payload = payload_path.read_text(encoding="utf-8").strip()
        if not target_id or not payload:
            continue
        if target_id in values and values[target_id] != payload:
            raise RuntimeError(f"Conflicting frozen payloads for Target {target_id}")
        values[target_id] = payload
        files[target_id] = str(payload_path)
    if not values:
        raise FileNotFoundError(f"No reusable payloads found for {dataset} under {root}")
    canonical = json.dumps(values, ensure_ascii=False, sort_keys=True).encode("utf-8")
    metadata = {
        "source_root": str(root.resolve()),
        "dataset": dataset,
        "payload_count": len(values),
        "target_payload_map_sha256": hashlib.sha256(canonical).hexdigest(),
        "source_files": files,
    }
    return values, metadata
