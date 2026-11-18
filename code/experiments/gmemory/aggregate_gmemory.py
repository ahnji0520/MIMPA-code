#!/usr/bin/env python3
"""Aggregate the six G-Memory main-table conditions across datasets."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
DATASETS = ('mmlu_pro', 'truthfulqa', 'medmcqa')
CONDITIONS = (
    'clean', 'mac_attack', 'autotransform', 'mimpa_wo_poisoned_record',
    'mimpa_wo_triggering', 'mimpa',
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding='utf-8'))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', default='seed42_n10_s10_k3_r2_t1_i3_embed')
    parser.add_argument('--model-slug', default='gpt_5_4_nano')
    parser.add_argument('--root', type=Path, default=ROOT / 'data/results/gmemory')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_root = args.root / args.run_id / args.model_slug
    rows: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        row: dict[str, Any] = {'condition': condition}
        for dataset in DATASETS:
            summary = read_json(run_root / dataset / 'summary.json')
            source_condition = condition
            if (
                condition == 'mimpa_wo_poisoned_record'
                and condition not in summary['conditions']
                and 'mimpa_wo_memory' in summary['conditions']
            ):
                source_condition = 'mimpa_wo_memory'
            metrics = summary['conditions'][source_condition]
            row[f'{dataset}_accuracy'] = metrics['accuracy']
            row[f'{dataset}_asr'] = metrics['asr']
            row[f'{dataset}_accuracy_n'] = metrics['accuracy_numerator']
            row[f'{dataset}_accuracy_d'] = metrics['accuracy_denominator']
            row[f'{dataset}_asr_n'] = metrics['asr_numerator']
            row[f'{dataset}_asr_d'] = metrics['asr_denominator_fixed_clean_correct']
        rows.append(row)

    payload = {
        'run_id': args.run_id,
        'model_slug': args.model_slug,
        'datasets': list(DATASETS),
        'conditions': list(CONDITIONS),
        'rows': rows,
    }
    (run_root / 'main_table.json').write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8'
    )
    columns = ['condition'] + [
        f'{dataset}_{metric}'
        for dataset in DATASETS
        for metric in ('accuracy', 'asr', 'accuracy_n', 'accuracy_d', 'asr_n', 'asr_d')
    ]
    with (run_root / 'main_table.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
