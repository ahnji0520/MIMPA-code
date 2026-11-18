#!/usr/bin/env python3
"""Build nested random evaluation samples from complete MMLU and TruthfulQA pools."""

from __future__ import annotations

import collections
import csv
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


DATA_ROOT = Path(__file__).resolve().parents[2] / 'data'
SEED = 42
MMLU_SOURCE = DATA_ROOT / 'mmlu/raw/mmlu_all_test.parquet'
TRUTHFULQA_SOURCE = DATA_ROOT / 'truthfulqa/raw/TruthfulQA.csv'
MMLU_URL = 'https://huggingface.co/datasets/cais/mmlu/resolve/main/all/test-00000-of-00001.parquet'


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n')


def mmlu_pool() -> list[dict[str, Any]]:
    rows = pq.read_table(MMLU_SOURCE).to_pylist()
    subjects = collections.Counter(row['subject'] for row in rows)
    assert len(rows) == 14042, f'Expected 14,042 MMLU test rows, got {len(rows)}'
    assert len(subjects) == 57, f'Expected 57 MMLU subjects, got {len(subjects)}'
    subject_index: dict[str, int] = collections.defaultdict(int)
    output = []
    for index, row in enumerate(rows):
        subject = row['subject']
        assert isinstance(row['question'], str) and len(row['choices']) == 4
        assert row['answer'] in range(4)
        output.append({
            **row,
            'source_id': f'mmlu_test_{index:05d}',
            'source_split': 'test',
            'source_index': index,
            'subject_index': subject_index[subject],
        })
        subject_index[subject] += 1
    return output


def truthfulqa_pool() -> list[dict[str, Any]]:
    with TRUTHFULQA_SOURCE.open(newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 790, f'Expected 790 TruthfulQA rows, got {len(rows)}'
    assert all(row['Question'] and row['Best Answer'] and row['Best Incorrect Answer'] for row in rows)
    return [
        {
            **row,
            'source_id': f'TruthfulQA/{index}',
            'source_index': index,
        }
        for index, row in enumerate(rows)
    ]


def summarize(name: str, rows: list[dict[str, Any]], parent: str, dataset: str) -> dict[str, Any]:
    summary: dict[str, Any] = {
        'name': name,
        'output': str(DATA_ROOT / dataset / 'sample' / name / 'targets.jsonl'),
        'parent': parent,
        'seed': SEED,
        'sampling_method': 'random.Random(seed).sample without replacement',
        'count': len(rows),
        'source_ids': [row['source_id'] for row in rows],
    }
    if dataset == 'mmlu':
        summary['subject_count'] = len({row['subject'] for row in rows})
        summary['by_subject'] = dict(sorted(collections.Counter(row['subject'] for row in rows).items()))
        summary['by_answer'] = dict(sorted(collections.Counter('ABCD'[row['answer']] for row in rows).items()))
    else:
        summary['by_category'] = dict(sorted(collections.Counter(row['Category'] for row in rows).items()))
        summary['by_type'] = dict(sorted(collections.Counter(row['Type'] for row in rows).items()))
    return summary


def build(dataset: str, pool: list[dict[str, Any]], source: Path, source_url: str | None) -> None:
    root = DATA_ROOT / dataset / 'sample'
    root.mkdir(parents=True, exist_ok=True)
    # The earlier five-subject/ten-case pilot used a different sample_10 schema.
    # Keep its case manifest and preserve its summary before replacing summary.json.
    prior_summary = root / 'sample_10/summary.json'
    prior_copy = root / 'sample_10/pilot_cases_summary.json'
    if prior_summary.exists() and not prior_copy.exists():
        prior_copy.write_bytes(prior_summary.read_bytes())

    sample_100 = random.Random(SEED).sample(pool, 100)
    for rank, row in enumerate(sample_100, 1):
        row['eval_id'] = f'{dataset}_eval_{rank:03d}'
        row['sample_100_rank'] = rank
    sample_10 = random.Random(SEED).sample(sample_100, 10)
    sample_5 = random.Random(SEED).sample(sample_10, 5)
    hierarchy = [
        ('sample_100', sample_100, f'full_pool_{len(pool)}'),
        ('sample_10', sample_10, 'sample_100'),
        ('sample_5', sample_5, 'sample_10'),
    ]
    summaries = {}
    for name, rows, parent in hierarchy:
        directory = root / name
        write_jsonl(directory / 'targets.jsonl', rows)
        summary = summarize(name, rows, parent, dataset)
        write_json(directory / 'summary.json', summary)
        summaries[name] = summary
    assert set(summaries['sample_5']['source_ids']) <= set(summaries['sample_10']['source_ids'])
    assert set(summaries['sample_10']['source_ids']) <= set(summaries['sample_100']['source_ids'])
    assert len({row['source_id'] for row in pool}) == len(pool)
    manifest = {
        'dataset': dataset,
        'source': str(source),
        'source_url': source_url,
        'source_sha256': digest(source),
        'source_pool_count': len(pool),
        'seed_at_every_stage': SEED,
        'sampling_scope': 'all rows pooled; no subject/category stratification',
        'hierarchy': f'full_pool_{len(pool)} -> sample_100 -> sample_10 -> sample_5',
        'nested_subset_check': True,
        'samples': {name: {
            'count': summary['count'],
            'targets': summary['output'],
            'summary': str(root / name / 'summary.json'),
        } for name, summary in summaries.items()},
        'legacy_pilot_cases': str(root / 'sample_10/cases.jsonl') if (root / 'sample_10/cases.jsonl').exists() else None,
        'legacy_pilot_summary': str(prior_copy) if prior_copy.exists() else None,
    }
    if dataset == 'mmlu':
        manifest['source_subject_count'] = len({row['subject'] for row in pool})
        manifest['source_subject_sizes'] = dict(sorted(collections.Counter(row['subject'] for row in pool).items()))
    write_json(root / 'manifest.json', manifest)
    (root / 'README.md').write_text(
        f'# {dataset} evaluation subsets\n\n'
        f'Full pool: {len(pool)} rows. A fresh random.Random(42) samples 100 rows '
        'without replacement from the pooled full dataset, then 10 from those 100, '
        'then 5 from those 10. There is no subject or category stratification.\n\n'
        'Each sample directory contains targets.jsonl with full source rows and '
        'summary.json with counts and ordered source IDs. manifest.json records '
        'the source checksum and paths.\n\n'
        'The pre-existing sample_10/cases.jsonl belongs to an earlier pilot and '
        'is preserved. Its former summary is pilot_cases_summary.json. New target '
        'experiments should read sample_10/targets.jsonl.\n',
        encoding='utf-8',
    )
    print(json.dumps({
        'dataset': dataset,
        'full_pool': len(pool),
        'sample_100': len(sample_100),
        'sample_10': len(sample_10),
        'sample_5': len(sample_5),
        'sample_100_subjects': summaries['sample_100'].get('subject_count'),
        'root': str(root),
    }, ensure_ascii=False))


if __name__ == '__main__':
    build('mmlu', mmlu_pool(), MMLU_SOURCE, MMLU_URL)
    build('truthfulqa', truthfulqa_pool(), TRUTHFULQA_SOURCE, None)
