#!/usr/bin/env python3
"""Attach MIMPA transcript-audit metrics to an existing combined Main table."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))

def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument("--main-table", type=Path, required=True)
    p.add_argument("--audit-summary", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args=p.parse_args()
    table=load(args.main_table)
    audit=load(args.audit_summary)
    for row in table["rows"]:
        dataset=row["dataset"]
        metrics=audit["datasets"][dataset]
        if metrics["eligible_clean_correct"] != row["fixed_asr_denominator_clean_correct"]:
            raise ValueError(f"{dataset}: clean-correct denominator mismatch")
        mimpa=row["conditions"]["mimpa"]
        if metrics["raw_mimpa_successes"] != mimpa["asr_numerator"]:
            raise ValueError(f"{dataset}: raw MIMPA numerator mismatch")
        row["mimpa_transcript_audit"] = metrics
    table["transcript_audit"] = {
        "summary_path": str(args.audit_summary),
        "definition": audit["definition"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(table, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output":str(args.output),"datasets":len(table["rows"])},ensure_ascii=False))

if __name__=="__main__":
    main()
