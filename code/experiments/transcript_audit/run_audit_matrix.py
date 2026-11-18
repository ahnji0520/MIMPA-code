#!/usr/bin/env python3
"""Plan or run the same MIMPA audit on shared-memory and G-Memory results."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = ROOT / 'code/experiments/transcript_audit'


def audit_command(source: Path, output: Path, args: argparse.Namespace) -> list[str]:
    return [
        sys.executable, str(HERE / 'run_mimpa_audit.py'),
        '--main-root', str(source), '--output-root', str(output),
        '--datasets', *args.datasets,
        '--model', args.model, '--effort', args.effort,
        '--repeats', str(args.repeats), '--workers', str(args.workers),
        '--threshold', str(args.threshold), '--max-output-tokens', str(args.max_output_tokens),
        '--api-key-env', args.api_key_env,
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--shared-root', type=Path)
    parser.add_argument('--gmemory-root', type=Path)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--datasets', nargs='+', default=['mmlu_pro', 'truthfulqa', 'medmcqa'])
    parser.add_argument('--model', default='gpt-5.4-mini')
    parser.add_argument('--effort', default='high')
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--workers', type=int, default=30)
    parser.add_argument('--threshold', type=int, default=50)
    parser.add_argument('--max-output-tokens', type=int, default=6000)
    parser.add_argument('--api-key-env', default='OPENAI_API_KEY')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    sources = {
        key: value for key, value in (
            ('shared_success_top3', args.shared_root), ('gmemory', args.gmemory_root)
        ) if value is not None
    }
    if not sources:
        parser.error('provide --shared-root and/or --gmemory-root')
    commands = {
        backend: audit_command(source, args.output_root / backend, args)
        for backend, source in sources.items()
    }
    print(json.dumps({
        'execute': args.execute,
        'sources': {k: str(v) for k, v in sources.items()},
        'commands': commands,
    }, ensure_ascii=False, indent=2))
    if not args.execute:
        return
    combined = {}
    for backend, command in commands.items():
        subprocess.run(command, cwd=ROOT, check=True)
        output = args.output_root / backend
        subprocess.run([
            sys.executable, str(HERE / 'aggregate_mimpa_audit.py'),
            '--audit-root', str(output),
        ], cwd=ROOT, check=True)
        combined[backend] = json.loads((output / 'summary.json').read_text(encoding='utf-8'))
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / 'summary_by_memory_backend.json').write_text(
        json.dumps(combined, ensure_ascii=False, indent=2) + '\n', encoding='utf-8'
    )


if __name__ == '__main__':
    main()
