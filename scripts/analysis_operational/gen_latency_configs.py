#!/usr/bin/env python3
"""Generate the same-hardware latency baseline configs.

The submitted paper measured E1 3,733 / E1-RF 2,143 / E3 11,098 / E4 4,350 ms across three
different serving paths (M5 Max MPS, Ollama 4-bit, NVIDIA vLLM), so the claimed 2.6x E3->E4
speedup is not established. These four configs pin every condition to one box, one engine
and one base checkpoint, so the latency column becomes internally comparable.

Changes vs the originals, and nothing else:
  * model.base   -> the local unsloth checkpoint (same weights the adapter was trained on)
  * model.adapter-> qlora_unsw_v2_clean for E4. The original E4_unsw.yaml still points at
                    qlora_unsw_v2, which was renamed to
                    qlora_unsw_v2_leaked_archive (val-split leak); the clean adapter is
                    canonical and per-sample identical.
  * data.subsample -> 200 for every condition, so the four rows share one evaluation draw.
                    Latency is a per-sample mean, so N=200 is ample; accuracy from these
                    runs is NOT a replacement for the N=1000 main table.

Usage: python3 gen_latency_configs.py <project_root> <remote_base_path>
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

CONDITIONS = ["E1_unsw", "E1_rf_context_unsw", "E3_unsw", "E4_unsw"]
CLEAN_ADAPTER = "project/models/qlora_unsw_v2_clean/adapter/"
N = 200


def main() -> int:
    root = Path(sys.argv[1]).resolve()
    base_path = sys.argv[2]
    for name in CONDITIONS:
        src = root / "configs" / "v2" / f"{name}.yaml"
        cfg = yaml.safe_load(src.read_text())
        stem = f"lat_{name}"

        cfg["model"]["base"] = base_path
        if cfg["model"].get("adapter"):
            cfg["model"]["adapter"] = CLEAN_ADAPTER
        cfg["data"]["subsample"] = N
        cfg["experiment"]["id"] = stem
        cfg["experiment"]["name"] = f"Latency baseline (same hardware): {name}"
        cfg["output"]["prefix"] = stem

        out = root / "configs" / "v2" / f"{stem}.yaml"
        out.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
        print(f"WROTE configs/v2/{stem}.yaml")
        print(f"      harness={cfg['harness']['enabled']} "
              f"rf_context={cfg['harness'].get('rf_context')} "
              f"adapter={cfg['model'].get('adapter')} N={cfg['data']['subsample']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
