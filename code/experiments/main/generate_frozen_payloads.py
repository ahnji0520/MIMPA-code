#!/usr/bin/env python3
"""Generate one frozen target-answer-only payload per Target for paired runs."""
from __future__ import annotations
import argparse, concurrent.futures, json, sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
CODE = ROOT / "code"
COMMON = CODE / "experiments" / "common"
for path in (CODE, COMMON):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from api_client import create_client
from target_answer_only import PROMPT_VERSION, generate_target_answer_only_plan, public_target

DATASETS = ("mmlu_pro", "truthfulqa", "medmcqa")

def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in value)[:96]

def complete_episode(path: Path, target_id: str) -> bool:
    summary, payload = path / "summary.json", path / "payload.txt"
    if not summary.exists() or not payload.exists():
        return False
    try:
        row = json.loads(summary.read_text(encoding="utf-8"))
    except Exception:
        return False
    return (row.get("complete") is True and str(row.get("target_id")) == target_id
            and row.get("payload_prompt_version") == PROMPT_VERSION
            and bool(payload.read_text(encoding="utf-8").strip()))

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=DATASETS)
    p.add_argument("--stream-name", default="stream_seed42_n100_s10")
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--shard-id", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--model", required=True)
    p.add_argument("--effort", default="off")
    p.add_argument("--backend", choices=("vllm", "openai"), default="vllm")
    p.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    if args.num_shards < 1 or not 0 <= args.shard_id < args.num_shards:
        raise ValueError("Require num-shards >= 1 and 0 <= shard-id < num-shards")
    stream_file = ROOT / "data" / args.dataset / "sample" / args.stream_name / "episodes.jsonl"
    selected = read_jsonl(stream_file)[:args.limit]
    episodes = [e for pos, e in enumerate(selected) if pos % args.num_shards == args.shard_id]
    manifest = {
        "dataset": args.dataset, "stream_name": args.stream_name, "model": args.model,
        "reasoning_effort": args.effort, "backend": args.backend,
        "payload_prompt_version": PROMPT_VERSION,
        "correct_answer_visible_to_payload_generator": False,
        "episode_count": len(episodes), "total_episode_count": len(selected),
        "shard_id": args.shard_id, "num_shards": args.num_shards,
        "episode_indices": [int(e["episode_index"]) for e in episodes],
        "output_root": str(args.output_root),
    }
    if args.dry_run:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return

    args.output_root.mkdir(parents=True, exist_ok=True)
    write_json(args.output_root / "manifest.json", manifest)
    client = create_client(backend=args.backend, audit_path=args.output_root / "api_calls.jsonl",
                           base_url=args.base_url, temperature=0.0, seed=42)

    def generate(episode: dict[str, Any]) -> dict[str, Any]:
        index = int(episode["episode_index"])
        target = episode["target"]
        target_id = str(target["task_id"])
        episode_dir = args.output_root / f"episode_{index:03d}_{safe_name(target_id)}"
        if complete_episode(episode_dir, target_id):
            return {"episode_index": index, "target_id": target_id, "status": "reused"}
        plan = generate_target_answer_only_plan(
            client, dataset=args.dataset, target=public_target(target),
            attack_answer=str(episode["attack_answer"]), model=args.model, effort=args.effort,
            purpose=f"{args.dataset}.frozen_payload.e{index}")
        episode_dir.mkdir(parents=True, exist_ok=True)
        (episode_dir / "payload.txt").write_text(str(plan["payload"]).rstrip() + "\n", encoding="utf-8")
        write_json(episode_dir / "plan.json", plan)
        summary = {
            "complete": True, "dataset": args.dataset, "episode_index": index,
            "target_id": target_id, "attack_answer": str(episode["attack_answer"]),
            "payload_prompt_version": PROMPT_VERSION, "answer_key_visible_to_generator": False,
        }
        write_json(episode_dir / "summary.json", summary)
        return {**summary, "status": "generated"}

    results: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(args.workers, len(episodes))) as pool:
        futures = [pool.submit(generate, episode) for episode in episodes]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda row: row["episode_index"])
    write_json(args.output_root / "generation_summary.json", {
        **manifest, "complete": len(results),
        "generated": sum(r["status"] == "generated" for r in results),
        "reused": sum(r["status"] == "reused" for r in results), "results": results,
    })
    print(json.dumps({
        "dataset": args.dataset, "shard_id": args.shard_id, "complete": len(results),
        "generated": sum(r["status"] == "generated" for r in results),
        "reused": sum(r["status"] == "reused" for r in results),
    }, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
