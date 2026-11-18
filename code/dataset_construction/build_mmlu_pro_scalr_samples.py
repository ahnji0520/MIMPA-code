#!/usr/bin/env python3
"""Build reproducible MMLU-Pro and LegalBench SCALR evaluation samples."""
from __future__ import annotations

import collections
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / 'data'
SEED = 42
SPECS = {
    'mmlu_pro': {
        'repo': 'TIGER-Lab/MMLU-Pro',
        'revision': 'b189ec765aa7ed75c8acfea42df31fdae71f97be',
        'split': 'test',
        'source': DATA_ROOT / 'mmlu_pro/raw/test.parquet',
        'url': 'https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro',
        'license': 'MIT',
    },
    'scalr': {
        'repo': 'nguha/legalbench',
        'config': 'scalr',
        'revision': 'daec8237410aa23e3faf4bc41ad8b3a7e1696826',
        'split': 'test',
        'source': DATA_ROOT / 'scalr/raw/test.tsv',
        'url': 'https://huggingface.co/datasets/nguha/legalbench',
        'license': 'CC-BY-4.0',
    },
}


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


def labels(n: int) -> list[str]:
    if not 2 <= n <= 10:
        raise ValueError(f'Unsupported option count: {n}')
    return [chr(ord('A') + i) for i in range(n)]


def mmlu_pro_pool() -> list[dict[str, Any]]:
    rows = pd.read_parquet(SPECS['mmlu_pro']['source']).to_dict(orient='records')
    assert len(rows) == 12032, len(rows)
    output = []
    for source_index, row in enumerate(rows):
        options = [str(value) for value in list(row['options'])]
        answer_index = int(row['answer_index'])
        answer_labels = labels(len(options))
        assert answer_index in range(len(options))
        assert str(row['answer']) == answer_labels[answer_index]
        output.append({
            'question_id': int(row['question_id']),
            'question': str(row['question']),
            'options': options,
            'answer': str(row['answer']),
            'answer_index': answer_index,
            'category': str(row['category']),
            'src': str(row['src']),
            'source_id': f"mmlu_pro_test_{int(row['question_id']):05d}",
            'source_index': source_index,
            'source_split': 'test',
        })
    assert len({row['source_id'] for row in output}) == len(output)
    assert all('cot_content' not in row for row in output)
    return output


def scalr_pool() -> list[dict[str, Any]]:
    rows = pd.read_csv(SPECS['scalr']['source'], sep='\t').to_dict(orient='records')
    assert len(rows) == 571, len(rows)
    output = []
    for source_index, row in enumerate(rows):
        options = [str(row[f'choice_{i}']) for i in range(5)]
        answer_index = int(row['answer'])
        assert answer_index in range(5)
        original_index = int(row['index'])
        output.append({
            'index': original_index,
            'question': str(row['question']),
            'options': options,
            'answer': labels(5)[answer_index],
            'answer_index': answer_index,
            'category': 'legal_reasoning',
            'source_id': f'scalr_test_{original_index:05d}',
            'source_index': source_index,
            'source_split': 'test',
        })
    assert len({row['source_id'] for row in output}) == len(output)
    return output


def summarize(dataset: str, name: str, rows: list[dict[str, Any]], parent: str) -> dict[str, Any]:
    summary: dict[str, Any] = {
        'name': name,
        'dataset': dataset,
        'output': str(DATA_ROOT / dataset / 'sample' / name / 'targets.jsonl'),
        'parent': parent,
        'seed': SEED,
        'sampling_method': 'random.Random(seed).sample without replacement',
        'count': len(rows),
        'source_ids': [row['source_id'] for row in rows],
        'by_answer': dict(sorted(collections.Counter(row['answer'] for row in rows).items())),
    }
    if dataset == 'mmlu_pro':
        summary['category_count'] = len({row['category'] for row in rows})
        summary['by_category'] = dict(sorted(collections.Counter(row['category'] for row in rows).items()))
        summary['by_option_count'] = dict(sorted(collections.Counter(len(row['options']) for row in rows).items()))
    return summary


def build(dataset: str, pool: list[dict[str, Any]]) -> None:
    spec = SPECS[dataset]
    sample_root = DATA_ROOT / dataset / 'sample'
    sample_root.mkdir(parents=True, exist_ok=True)
    sample_100 = random.Random(SEED).sample(pool, 100)
    for rank, row in enumerate(sample_100, 1):
        row['eval_id'] = f'{dataset}_eval_{rank:03d}'
        row['sample_100_rank'] = rank
    sample_10 = random.Random(SEED).sample(sample_100, 10)
    hierarchy = [
        ('sample_100', sample_100, f'full_pool_{len(pool)}'),
        ('sample_10', sample_10, 'sample_100'),
    ]
    summaries = {}
    for name, rows, parent in hierarchy:
        out = sample_root / name
        write_jsonl(out / 'targets.jsonl', rows)
        summaries[name] = summarize(dataset, name, rows, parent)
        write_json(out / 'summary.json', summaries[name])
    assert set(summaries['sample_10']['source_ids']) <= set(summaries['sample_100']['source_ids'])
    raw_root = DATA_ROOT / dataset / 'raw'
    raw_files = {
        path.name: {'sha256': digest(path), 'bytes': path.stat().st_size}
        for path in sorted(raw_root.iterdir()) if path.is_file() and path.name != 'manifest.json'
    }
    write_json(raw_root / 'manifest.json', {
        'dataset': dataset,
        'repository': spec['repo'],
        'config': spec.get('config'),
        'revision': spec['revision'],
        'source_url': spec['url'],
        'license': spec['license'],
        'evaluation_split': spec['split'],
        'evaluation_pool_count': len(pool),
        'raw_files': raw_files,
        'notes': (
            'cot_content is preserved only in the MMLU-Pro raw parquet and is excluded '
            'from all sample and stream records.' if dataset == 'mmlu_pro' else
            'The current official Hugging Face scalr config exposes all 571 rows as test.'
        ),
    })
    write_json(sample_root / 'manifest.json', {
        'dataset': dataset,
        'source': str(spec['source']),
        'source_sha256': digest(spec['source']),
        'source_pool_count': len(pool),
        'seed_at_every_stage': SEED,
        'sampling_scope': 'all test rows pooled; no category stratification',
        'hierarchy': f'full_pool_{len(pool)} -> sample_100 -> sample_10',
        'nested_subset_check': True,
        'answer_and_choice_validation': True,
        'samples': {
            name: {
                'count': summary['count'],
                'targets': summary['output'],
                'summary': str(sample_root / name / 'summary.json'),
            }
            for name, summary in summaries.items()
        },
    })
    (sample_root / 'README.md').write_text(
        f'# {dataset} evaluation subsets\n\n'
        f'The official {spec["split"]} split contains {len(pool)} rows. A fresh '
        'random.Random(42) samples 100 rows without replacement from the full pool, '
        'then 10 rows from sample_100. Sampling is not stratified.\n\n'
        'Each sample directory contains targets.jsonl and summary.json. The top-level '
        'manifest records source provenance, checksum, and nesting validation.\n',
        encoding='utf-8',
    )
    print(json.dumps({
        'dataset': dataset,
        'full_pool': len(pool),
        'sample_100': len(sample_100),
        'sample_10': len(sample_10),
        'sample_100_categories': summaries['sample_100'].get('category_count'),
        'root': str(sample_root),
    }, ensure_ascii=False))


if __name__ == '__main__':
    build('mmlu_pro', mmlu_pro_pool())
    build('scalr', scalr_pool())
