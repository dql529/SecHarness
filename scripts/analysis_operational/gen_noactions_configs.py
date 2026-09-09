#!/usr/bin/env python3
"""Generate the −A (action-space) ablation configs for Table 7's four capacities.

Each new config is its base 'full' config plus harness.actions disabling escalate and
log_decision. classify stays (constitutive). Nothing else changes, so the new column is
comparable with the existing −K/−O/−P/−T columns of the same row.

Usage: python3 gen_noactions_configs.py <project_root>
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

# (base config, new config, id/prefix stem) — one per Table 7 row
PAIRS = [
    ("configs/v2/E4_ablation_full_unsw.yaml",
     "configs/v2/E4_ablation_noActions_unsw.yaml", "E4_ablation_noActions_unsw"),
    ("configs/v2/E3_7B_ablation_noKnowledge_unsw.yaml",
     "configs/v2/E3_7B_ablation_noActions_unsw.yaml", "E3_7B_ablation_noActions_unsw"),
    ("configs/v2_tdsc/tau/tau_13B_full_unsw.yaml",
     "configs/v2_tdsc/tau/tau_13B_noActions_unsw.yaml", "tau_13B_noActions_unsw"),
    ("configs/v2_tdsc/tau/tau_32B_full_unsw.yaml",
     "configs/v2_tdsc/tau/tau_32B_noActions_unsw.yaml", "tau_32B_noActions_unsw"),
]


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
    for base_rel, out_rel, stem in PAIRS:
        base = root / base_rel
        if not base.exists():
            print(f"SKIP (no base): {base_rel}")
            continue
        cfg = yaml.safe_load(base.read_text())

        # The 7B row's base is a noKnowledge config (no 'full' exists); restore knowledge
        # so the only difference from that row's full condition is the action space.
        if "E3_7B" in stem:
            cfg["harness"].setdefault("tools", {})["load_knowledge"] = True

        cfg["experiment"]["id"] = stem
        cfg["experiment"]["name"] = f"Ablation: noActions ({stem.split('_')[0]})"
        cfg["harness"]["actions"] = {"escalate": False, "log_decision": False}
        cfg["output"]["prefix"] = stem
        cfg.setdefault("data", {})["subsample"] = 200   # Table 7 uses N=200 per condition

        out = root / out_rel
        out.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
        print(f"WROTE {out_rel}")
        print(f"      model={cfg['model']['base']}  subsample={cfg['data']['subsample']}  "
              f"actions={cfg['harness']['actions']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
