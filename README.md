# SecHarness — code mirror of the supplementary archive

This repository mirrors the code, configuration and analysis scripts of the supplementary
archive that accompanies the manuscript *Forecasting Deployment-Time Blind Spots in
Language-Model Intrusion Detection Agents: A Data-Geometric Coverage Statistic and the Relay
Condition That Carries It to the Agent* (Expert Systems, under review). Directories `src/`,
`configs/`, `scripts/` and `tests/` are byte-identical to the archive.

The evidence the manuscript's tables are computed from — the per-sample audit logs, the results
tables and the seed-replication provenance sidecars and SHA-256 manifests (2.2 GB uncompressed) —
is not in this repository: it accompanies the manuscript as the supplementary archive
`SecHarness_supplementary.zip`, whose layout this repository mirrors, and is released with the
paper. `MANIFEST.md` inventories that archive (every per-sample log with its record count and
hash); unzip the archive over this directory and every relative path resolves.

Directory tags such as `v2_tdsc` and the `13B` prefix of the 14B configurations are legacy
internal names kept so that archive paths stay stable; see `MANIFEST.md`.

Licence: MIT (see `LICENSE`).

---

# SecHarness supplementary archive

Anonymised code, configuration, evidence and analysis for the manuscript. Layout mirrors the
project the experiments ran in, so every relative path in the scripts resolves from this
directory.

- `src/` -- the harness and agent implementation (tools, permission layer, audit trail,
  serialisation, agent loop).
- `configs/` -- every experiment configuration for every condition reported.
- `logs/` -- per-sample `AuditRecordV2` JSONL audit logs for the conditions reported,
  including both ML-only baselines. `MANIFEST.md` inventories every archived log (record
  counts and hashes) and lists the conditions that ship summary files only because their
  per-sample logs did not survive:
  * Qwen2.5-14B tool-withdrawal (-T), the earlier run, superseded (file prefix tau_13B is the legacy tag of the 14B configuration): `results/tables/v2_tdsc/tau_13B_noTools_unsw_EARLIER_RUN_summary.json`; feeds no reported figure (superseded 2026-09-09: the configuration was re-run with five sampling seeds, Supplementary Section S27, and this run was set aside as an outlier; the summary file is kept for the record).
  The analysis scripts regenerate the reported tables from the archived logs without GPU
  access. For anonymisation, absolute filesystem paths inside records
  (the fine-tuned conditions' `agent_config.adapter` field) are rewritten to
  repository-relative form; no measurement field is modified.
- `scripts/` -- run scripts per condition and the analysis scripts producing each table.
  `scripts/analysis_operational/d3_operational_quality.py` is the gated script behind the
  operational-quality section: it refuses to emit numbers unless its ten instrument checks
  pass. Its companion `d3_risk_coverage_curve.py` draws the risk--coverage figure
  (`python3 scripts/analysis_operational/d3_risk_coverage_curve.py . --out fig_risk_coverage.pdf`).
  `replay_fix_impact.py` reproduces the post-replay tool-ablation values.
  `scripts/candidate_b/` reproduces the coverage-forecasting study end to end.
- `tests/` -- unit tests for the harness components.
- `results/tables/` -- published summary tables (files over 20 MB are omitted: the per-fold
  random-forest pickles and per-seed sample dumps are regenerated deterministically by
  `scripts/candidate_b/run_coverage_zd_stageA.py` and companions from the public datasets
  with the seeds fixed in the scripts).

Datasets are not bundled: CIC-IDS2017 and UNSW-NB15 are public, and preprocessing lives in
`src/data`. Model weights are pulled from public checkpoints named in the configs.
