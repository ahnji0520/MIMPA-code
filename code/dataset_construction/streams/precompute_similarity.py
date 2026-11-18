#!/usr/bin/env python3
"""Precompute Qwen3 embedding similarities for every main episode."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = ROOT / "data"
DATASETS = ("mmlu", "truthfulqa", "medmcqa", "mmlu_pro", "scalr")
DEFAULT_MODEL = os.environ.get("MIMPA_EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-8B")
QUERY_INSTRUCTION = (
    "Given a current multiple-choice question, retrieve prior solved questions "
    "that are semantically useful for answering it"
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def query_text(text: str) -> str:
    return f"Instruct: {QUERY_INSTRUCTION}\nQuery:{text}"


def normalize(vector: list[float]) -> list[float]:
    norm = sum(value * value for value in vector) ** 0.5
    if not norm:
        raise ValueError("Embedding with zero norm")
    return [value / norm for value in vector]


def dot(left: list[float], right: list[float]) -> float:
    return float(sum(a * b for a, b in zip(left, right)))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--stream-name", default="stream_seed42_n100_s10")
    p.add_argument("--gpu", default="4")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--max-model-len", type=int, default=2048)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    args = p.parse_args()

    # CUDA_VISIBLE_DEVICES is set before vLLM/torch import; the selected physical GPU is then cuda:0.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    from vllm import LLM  # pylint: disable=import-outside-toplevel

    episode_sets: dict[str, list[dict[str, Any]]] = {}
    all_texts: list[str] = []
    for dataset in args.datasets:
        path = DATA_ROOT / dataset / "sample" / args.stream_name / "episodes.jsonl"
        episodes = read_jsonl(path)
        episode_sets[dataset] = episodes
        for episode in episodes:
            tasks = episode["pre_target_tasks"] + [episode["target"]]
            all_texts.extend(task["retrieval_text"] for task in tasks)

    unique_texts = list(dict.fromkeys(all_texts))
    model = LLM(
        model=args.model,
        runner="pooling",
        convert="embed",
        trust_remote_code=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )
    document_outputs = model.embed(unique_texts, use_tqdm=True)
    query_outputs = model.embed([query_text(text) for text in unique_texts], use_tqdm=True)
    document_vectors = {
        text: normalize(list(output.outputs.embedding))
        for text, output in zip(unique_texts, document_outputs)
    }
    query_vectors = {
        text: normalize(list(output.outputs.embedding))
        for text, output in zip(unique_texts, query_outputs)
    }

    for dataset, episodes in episode_sets.items():
        rows = []
        for episode in episodes:
            tasks = episode["pre_target_tasks"] + [episode["target"]]
            ids = [task["task_id"] for task in tasks]
            texts = [task["retrieval_text"] for task in tasks]
            matrix = [
                [round(dot(query_vectors[qtext], document_vectors[dtext]), 8) for dtext in texts]
                for qtext in texts
            ]
            rankings = []
            for query_index, query_id in enumerate(ids):
                eligible_indices = list(range(query_index))
                ranked = sorted(eligible_indices, key=lambda index: (-matrix[query_index][index], index))
                rankings.append({
                    "query_task_id": query_id,
                    "query_position": query_index + 1 if query_index < 10 else "target",
                    "ranked_predecessors": [
                        {"task_id": ids[index], "position": index + 1, "similarity": matrix[query_index][index]}
                        for index in ranked
                    ],
                })
            rows.append({
                "schema_version": 1,
                "dataset": dataset,
                "episode_index": episode["episode_index"],
                "task_ids": ids,
                "query_document_cosine_matrix": matrix,
                "rankings": rankings,
            })

        out = DATA_ROOT / dataset / "sample" / args.stream_name
        cache_name = "similarities_qwen3_embedding_8b.jsonl"
        write_jsonl(out / cache_name, rows)
        write_json(out / "similarity_manifest.json", {
            "schema_version": 1,
            "dataset": dataset,
            "seed": args.seed,
            "stream_name": args.stream_name,
            "physical_gpu": str(args.gpu),
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "model": args.model,
            "backend": "vllm_embed",
            "query_instruction": QUERY_INSTRUCTION,
            "query_format": "Instruct: {instruction}\\nQuery:{task_text}",
            "document_format": "raw task retrieval_text",
            "metric": "cosine_similarity",
            "cache_file": cache_name,
            "runtime_rule": "filter ranked predecessors to successful stored trajectories, then take top 3",
            "attacker_access_to_scores": False,
        })
        print(f"{dataset}: wrote {len(rows)} episode caches to {out / cache_name}")


if __name__ == "__main__":
    main()
