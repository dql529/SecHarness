#!/usr/bin/env python3
"""
Derive E4 (NoHarness) metrics from E5 audit log.

Since E4 and E5 use the same Alpha and Beta agents, E4's results can be
derived by using only Alpha's verdict (ignoring consensus) from E5's audit log.
"""

import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT.parent))

from project.src.harness.metrics import (
    export_all_metrics, compute_binary_metrics, compute_zeroday_rate,
    compute_harness_metrics, compute_efficiency_metrics,
)
from project.src.harness.observability import AuditRecord as _AuditRecord


def derive_e4_records(e5_audit_path: Path) -> list:
    """Re-derive E4 records: Alpha verdict is final, no consensus."""
    records = []
    with open(e5_audit_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            # Override consensus to alpha-only
            alpha = d["alpha_verdict"]
            beta = d["beta_verdict"]
            if alpha["verdict"] == beta["verdict"]:
                ctype = "agreement"
            else:
                ctype = "alpha_only"

            d["consensus_result"]["final_verdict"] = alpha["verdict"]
            d["consensus_result"]["final_attack_type"] = alpha.get("attack_type")
            d["consensus_result"]["consensus_type"] = ctype
            d["consensus_result"]["combined_confidence"] = alpha["confidence"]

            # Recompute detection_correct
            gt = d["ground_truth_label"]
            gt_attack = gt.lower() not in ("normal", "benign")
            pred_attack = alpha["verdict"] == "attack"
            d["detection_correct"] = (gt_attack == pred_attack)

            # Convert to AuditRecord
            record = _AuditRecord.from_dict(d)
            records.append(record)
    return records


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--e5-audit", type=str, required=True)
    ap.add_argument("--zeroday-classes", type=str, required=True)
    ap.add_argument("--output-prefix", type=str, default="E4_cic_full")
    args = ap.parse_args()

    zeroday_classes = [c.strip() for c in args.zeroday_classes.split(",")]
    e5_path = Path(args.e5_audit)

    print(f"Deriving E4 from E5 audit: {e5_path}")
    records = derive_e4_records(e5_path)
    print(f"  Total records: {len(records)}")

    # Export metrics
    results_dir = PROJECT / "results" / "tables"
    dataset_tag = "_".join(args.output_prefix.split("_")[1:])
    exported = export_all_metrics(
        records=records,
        experiment_id="E4",
        dataset=dataset_tag,
        output_dir=results_dir,
        zeroday_classes=zeroday_classes,
    )
    for group, path in exported.items():
        print(f"  Exported {group} → {path}")

    # Summary
    bm = compute_binary_metrics(records)
    zd = compute_zeroday_rate(records, zeroday_classes)
    print(f"\n=== E4 (derived from E5) ===")
    print(f"Accuracy={bm['accuracy']:.4f}  F1={bm['f1']:.4f}  FPR={bm['fpr']:.4f}")
    print(f"Zero-day: {zd['zeroday_detected']}/{zd['zeroday_total']} = {zd['zeroday_detection_rate']:.4f}")

    # Save audit log
    audit_out = PROJECT / "logs" / f"{args.output_prefix}_audit.jsonl"
    with open(audit_out, "w") as f:
        for r in records:
            f.write(r.to_json() + "\n")
    print(f"E4 audit saved: {audit_out}")


if __name__ == "__main__":
    main()
