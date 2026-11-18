"""Strict reuse of clean stream and clean Target artifacts."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_dataset_root(root: Path, dataset: str) -> Path:
    nested = root / dataset
    if nested.is_dir():
        return nested
    if root.name == dataset and root.is_dir():
        return root
    return nested


def _episode_candidates(root: Path, dataset: str, target_id: str) -> list[Path]:
    base = resolve_dataset_root(root, dataset)
    safe = target_id.replace("/", "_")
    candidates = list(base.glob(f"episode_*_{safe}"))
    candidates += list((base / "shards").glob(f"shard_*/episode_*_{safe}"))
    return candidates


def _compatible_episode(
    episode_dir: Path, *, target_id: str, episode_index: int,
) -> dict[str, Any] | None:
    summary_path = episode_dir / "summary.json"
    if not summary_path.exists():
        return None
    summary = _read(summary_path)
    if not (
        summary.get("complete")
        and str(summary.get("target_id")) == str(target_id)
        and int(summary.get("episode_index", -1)) == int(episode_index)
    ):
        return None
    return summary



def _manifest_for_episode(episode_dir: Path) -> dict[str, Any]:
    for parent in (episode_dir.parent, episode_dir.parent.parent, episode_dir.parent.parent.parent):
        path = parent / "manifest.json"
        if path.exists():
            return _read(path)
    return {}


def _config_matches(episode_dir: Path, expected: dict[str, Any] | None) -> bool:
    if not expected:
        return True
    manifest = _manifest_for_episode(episode_dir)
    return bool(manifest) and all(manifest.get(key) == value for key, value in expected.items())

def load_clean_target(
    root: Path | None, *, dataset: str, target_id: str, episode_index: int,
    model: str, effort: str, expected_memory_config: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], str] | None:
    if root is None:
        return None
    valid: list[tuple[dict[str, Any], Path]] = []
    for episode_dir in _episode_candidates(root, dataset, target_id):
        summary = _compatible_episode(
            episode_dir, target_id=target_id, episode_index=episode_index
        )
        log_path = episode_dir / "target_clean.json"
        if (
            summary is None
            or not log_path.exists()
            or not _config_matches(episode_dir, expected_memory_config)
        ):
            continue
        log = _read(log_path)
        if str(log.get("model")) != str(model):
            continue
        if str(log.get("reasoning_effort")) != str(effort):
            continue
        valid.append((log, log_path))
    if not valid:
        raise FileNotFoundError(
            f"No compatible clean Target result for {dataset}/{target_id} "
            f"episode {episode_index} under {root}"
        )
    if len(valid) > 1:
        raise RuntimeError(
            f"Ambiguous clean Target reuse for {dataset}/{target_id}: "
            f"{[str(path) for _, path in valid]}"
        )
    log, path = valid[0]
    return log, str(path)


def _task_log(directory: Path, position: int) -> Path:
    matches = sorted(
        path
        for path in directory.glob(f"task_{position:02d}_*.json")
        if not path.stem.endswith("_retrieval")
    )
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one clean task log at position {position} in {directory}; "
            f"found {len(matches)}"
        )
    return matches[0]


def load_clean_stream(
    root: Path | None, *, dataset: str, target_id: str, episode_index: int,
    tasks: list[dict[str, Any]], model: str, effort: str, memory_backend: str,
    expected_memory_config: dict[str, Any] | None = None,
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]] | None:
    if root is None:
        return None
    valid: list[tuple[dict[int, dict[str, Any]], Path]] = []
    for episode_dir in _episode_candidates(root, dataset, target_id):
        summary = _compatible_episode(
            episode_dir, target_id=target_id, episode_index=episode_index
        )
        if summary is None or not _config_matches(episode_dir, expected_memory_config):
            continue
        retrieval = summary.get("retrieval", {})
        source_backend = (
            "gmemory" if "strict_gate_open" in retrieval else "shared_success_top3"
        )
        if source_backend != memory_backend:
            continue
        logs: dict[int, dict[str, Any]] = {}
        compatible = True
        for position, task in enumerate(tasks, 1):
            try:
                path = _task_log(episode_dir / "clean_stream", position)
            except RuntimeError:
                compatible = False
                break
            log = _read(path)
            if (
                str(log.get("task", {}).get("task_id")) != str(task["task_id"])
                or str(log.get("model")) != str(model)
                or str(log.get("reasoning_effort")) != str(effort)
                or log.get("stage") != "honest"
            ):
                compatible = False
                break
            logs[position] = log
        if compatible and len(logs) == len(tasks):
            valid.append((logs, episode_dir))
    if not valid:
        raise FileNotFoundError(
            f"No compatible {memory_backend} clean stream for "
            f"{dataset}/{target_id} episode {episode_index} under {root}"
        )
    if len(valid) > 1:
        raise RuntimeError(
            f"Ambiguous clean-stream reuse for {dataset}/{target_id}: "
            f"{[str(path) for _, path in valid]}"
        )
    logs, episode_dir = valid[0]
    return logs, {
        "source_episode": str(episode_dir),
        "memory_backend": memory_backend,
        "task_count": len(logs),
        "model": model,
        "reasoning_effort": effort,
        "gmemory_manager_replay_required": memory_backend == "gmemory",
    }
