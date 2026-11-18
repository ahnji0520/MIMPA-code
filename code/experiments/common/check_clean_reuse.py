#!/usr/bin/env python3
"""Validate reusable clean streams and Target logs without making model calls."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
COMMON = ROOT / 'code/experiments/common'
if str(COMMON) not in sys.path:
    sys.path.insert(0, str(COMMON))

from clean_reuse import load_clean_stream, load_clean_target  # noqa: E402

DATASETS = ('mmlu_pro', 'truthfulqa', 'medmcqa')


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def expected_config(backend: str) -> dict[str, Any]:
    if backend == 'shared_success_top3':
        return {'retrieval_top_k': 3, 'stream_length': 10}
    return {
        'retrieval_top_k': 1,
        'insight_top_k': 3,
        'retrieval_threshold': 0.0,
        'hop': 1,
        'start_insights_threshold': 5,
        'rounds_per_insights': 5,
        'insights_point_num': 5,
        'manager_prompt_mode': 'qa_adapted',
        'stream_length': 10,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-root', type=Path, required=True,
                        help='Model-level result root containing dataset directories')
    parser.add_argument('--backend', choices=('shared_success_top3', 'gmemory'), required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--effort', required=True)
    parser.add_argument('--stream-name', default='stream_seed42_n100_s10')
    parser.add_argument('--datasets', nargs='+', choices=DATASETS, default=list(DATASETS))
    parser.add_argument('--limit', type=int, default=100)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()

    report: dict[str, Any] = {
        'source_root': str(args.source_root),
        'backend': args.backend,
        'model': args.model,
        'reasoning_effort': args.effort,
        'stream_name': args.stream_name,
        'expected_memory_config': expected_config(args.backend),
        'datasets': {},
    }
    for dataset in args.datasets:
        episodes_path = ROOT / 'data' / dataset / 'sample' / args.stream_name / 'episodes.jsonl'
        episodes = read_jsonl(episodes_path)[:args.limit]
        stream_ok = 0
        target_ok = 0
        errors: list[dict[str, Any]] = []
        for episode in episodes:
            index = int(episode['episode_index'])
            target_id = str(episode['target']['task_id'])
            try:
                load_clean_stream(
                    args.source_root,
                    dataset=dataset,
                    target_id=target_id,
                    episode_index=index,
                    tasks=episode['pre_target_tasks'],
                    model=args.model,
                    effort=args.effort,
                    memory_backend=args.backend,
                    expected_memory_config=expected_config(args.backend),
                )
                stream_ok += 1
            except Exception as exc:
                errors.append({'episode_index': index, 'target_id': target_id,
                               'artifact': 'clean_stream', 'error': str(exc)})
            try:
                load_clean_target(
                    args.source_root,
                    dataset=dataset,
                    target_id=target_id,
                    episode_index=index,
                    model=args.model,
                    effort=args.effort,
                    expected_memory_config=expected_config(args.backend),
                )
                target_ok += 1
            except Exception as exc:
                errors.append({'episode_index': index, 'target_id': target_id,
                               'artifact': 'clean_target', 'error': str(exc)})
        report['datasets'][dataset] = {
            'requested_episodes': len(episodes),
            'reusable_clean_streams': stream_ok,
            'reusable_clean_targets': target_ok,
            'fully_reusable_episodes': len(episodes) - len({e['episode_index'] for e in errors}),
            'errors': errors,
        }

    text = json.dumps(report, ensure_ascii=False, indent=2) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding='utf-8')
    print(text, end='')


if __name__ == '__main__':
    main()
