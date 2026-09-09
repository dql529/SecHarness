"""Step 4: mechanical label-hygiene audit across every audit log. CPU only, no GPU, no re-inference.

Why this script exists: filenames in logs/v2 and logs/v2_tdsc encode a NOMINAL model size
(E3_8B_..., tau_13B_...) that was chosen when the experiment was planned. The three other places
that also carry a model identity -- the YAML config's model.base, the per-record
agent_config.model actually written by the harness, and (for the tool-channel arms) whether a
check_anomaly step is even functional -- are the ground truth of what really ran. Anything that
cites a result by its filename/experiment-id alone can silently name the wrong model. This script
never assumes the filename is right: it reads agent_config.model from every line of every log,
reads model.base from the matching config where one can be found, and reports every place the
three disagree, plus which "tool-channel" arms are tool-channel in name only (the noRF/noTools
ablations still log a check_anomaly step -- it just always errors with "Unknown tool").

Usage: python3 step4_label_hygiene.py [--project DIR]
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# directory layout
# ---------------------------------------------------------------------------

# (display label, path relative to --project, is_backup_or_secondary)
GROUPS: list[tuple[str, str, bool]] = [
    ("v2", "logs/v2", False),
    ("v2_tdsc", "logs/v2_tdsc", False),
    ("v2/sub200_backup", "logs/v2/sub200_backup", True),
    ("v2/backup_wrong_cic_run", "logs/v2/backup_wrong_cic_run", True),
    ("v2_tdsc/audit_jsonl_archive", "logs/v2_tdsc/audit_jsonl_archive", True),
    ("multiseed", "logs/multiseed", True),
]

# ---------------------------------------------------------------------------
# filename-side parsing
# ---------------------------------------------------------------------------

_FN_SIZE_TOKEN = re.compile(r"^\d+B$")
_SUBSAMPLE_SUFFIX = re.compile(r"_(sub\d+|pilot\d+)$")
_FAMILY_PREFIXES = ("llama", "qwen", "gemma")


def filename_implied_size(stem: str) -> str:
    """Model size implied by the filename, read off a whole underscore-delimited '<digits>B'
    token (E3_8B_... -> '8B', E1_rf_context_12B_... -> '12B', E3_32B_local_... -> '32B' -- the
    trailing '_local'/'_ablation'/etc. is a separate token so it never interferes). 'unspecified'
    if no token in the filename matches -- true for every default-model E1-E4 unsw/cic run, only
    the scaling-sweep arms (E1_rf_context_*, E3_*, E3_noRF_*, tau_*) name a size at all."""
    for tok in stem.split("_"):
        if _FN_SIZE_TOKEN.match(tok):
            return tok
    return "unspecified"


def filename_implied_family(stem: str) -> str:
    """Model family implied by the filename. Real (not hard-coded) check: split on '_'/'-' and
    look for a token that STARTS WITH a known family name -- startswith, not substring, so
    'E1_..._8B_ollama_...' doesn't false-positive on 'llama' inside 'ollama'. 'unspecified' if
    nothing matches."""
    for tok in re.split(r"[_\-]", stem.lower()):
        for fam in _FAMILY_PREFIXES:
            if tok.startswith(fam):
                return fam.capitalize()
    return "unspecified"


# ---------------------------------------------------------------------------
# recorded-model-side parsing (agent_config.model)
# ---------------------------------------------------------------------------

_REC_SIZE = re.compile(r"(?i)\b(e)?(\d+)b\b")
_REC_LLAMA = re.compile(r"llama-?3\.([123])")


def recorded_size(model: str) -> str:
    """Size actually recorded in agent_config.model. 'qwen2.5:14b'/'...Qwen2.5-14B-Instruct' ->
    '14B'. 'gemma4:e4b' -> 'E4B', kept distinct from a plain '4B': the 'e' is Google's own
    "effective parameter count" marker on Gemma-3n-style tags, not a raw dense size, so folding it
    into '4B' would assert something the tag itself doesn't claim. 'n/a' if the string has no
    <digits>B pattern at all (e.g. the RandomForest baseline)."""
    m = _REC_SIZE.search(model)
    if not m:
        return "n/a"
    e_prefix, digits = m.groups()
    return f"{'E' if e_prefix else ''}{digits}B"


def recorded_family(model: str) -> str:
    """Family actually recorded in agent_config.model."""
    low = model.lower()
    if "qwen2.5" in low:
        return "Qwen2.5"
    m = _REC_LLAMA.search(low)
    if m:
        return f"Llama-3.{m.group(1)}"
    if "gemma4" in low:
        return "Gemma4"
    return f"unrecognized({model[:40]})"


# ---------------------------------------------------------------------------
# check_anomaly step classification
# ---------------------------------------------------------------------------


def check_anomaly_ok(step: dict) -> bool:
    """A check_anomaly step is functional evidence iff its output is not the harness's
    '{"error": "Unknown tool: check_anomaly"}' guard (emitted by the noRF/noTools ablations, which
    disable the tool but the agent still attempts to call it -- so the step exists in tool_chain
    either way, only the output tells you whether the channel was actually live). Every other
    output shape seen in this corpus -- the standard {"prediction":...} form and the E3_degraded
    arm's {"anomaly_detected":...} form -- counts as a real (successful) tool call."""
    out = step.get("output")
    parsed = out
    if isinstance(out, str):
        try:
            parsed = json.loads(out)
        except json.JSONDecodeError:
            return True  # non-JSON string output is not the known error shape
    return not (isinstance(parsed, dict) and "error" in parsed)


# ---------------------------------------------------------------------------
# per-file scan
# ---------------------------------------------------------------------------


@dataclass
class FileStats:
    path: Path
    group: str
    is_backup: bool
    n_records: int = 0
    bad_json: int = 0
    n_no_model_field: int = 0
    models: set[str] = field(default_factory=set)
    check_anomaly_total: int = 0
    check_anomaly_ok: int = 0
    check_anomaly_unavailable: int = 0

    @property
    def has_check_anomaly_step(self) -> bool:
        return self.check_anomaly_total > 0


def scan_file(
    path: Path,
    group: str,
    is_backup: bool,
    model_records: dict[str, int],
    model_files: dict[str, set[str]],
    model_check: dict[str, dict[str, int]],
) -> FileStats:
    """Single pass over one log file. Also accumulates the corpus-wide per-recorded-model
    aggregates needed for question 5(c) (attributed per RECORD, not per file, so it stays correct
    even if a file ever turns out to be MIXED)."""
    st = FileStats(path=path, group=group, is_backup=is_backup)
    text = path.read_text(errors="replace") if path.stat().st_size else ""
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        st.n_records += 1
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            st.bad_json += 1
            continue
        ac = rec.get("agent_config")
        model = ac.get("model") if isinstance(ac, dict) else None
        if not model:
            st.n_no_model_field += 1
            continue
        st.models.add(model)
        model_records[model] += 1
        model_files[model].add(f"{group}/{path.name}")

        rec_ok = rec_unavail = 0
        for step in rec.get("tool_chain") or []:
            if not isinstance(step, dict) or step.get("tool") != "check_anomaly":
                continue
            st.check_anomaly_total += 1
            if check_anomaly_ok(step):
                st.check_anomaly_ok += 1
                rec_ok += 1
            else:
                st.check_anomaly_unavailable += 1
                rec_unavail += 1
        model_check[model]["ok"] += rec_ok
        model_check[model]["unavailable"] += rec_unavail
    return st


def fmt_models(st: FileStats) -> str:
    if st.models:
        if len(st.models) == 1:
            return next(iter(st.models))
        return f"MIXED({len(st.models)}): " + ", ".join(sorted(st.models))
    if st.n_records == 0:
        return "EMPTY FILE (0 records)"
    return "NO agent_config.model FIELD"


# ---------------------------------------------------------------------------
# config resolution (task 6)
# ---------------------------------------------------------------------------

_API_PREFIX = re.compile(r"^api://[^/]+/")
NO_MODEL_BLOCK = object()  # sentinel: config has no top-level 'model:' key at all


def config_candidate_stem(log_stem: str) -> str:
    """log filename (minus .jsonl) -> the experiment id used as the config filename, e.g.
    'E3_8B_unsw_pilot10_audit' -> 'E3_8B_unsw', 'tau_13B_full_unsw_sub200_audit' ->
    'tau_13B_full_unsw'. Strips, in order: a trailing _hallucinated, a trailing _audit, then any
    number of trailing _sub<N> / _pilot<N> tokens."""
    s = log_stem
    if s.endswith("_hallucinated"):
        s = s[: -len("_hallucinated")]
    if s.endswith("_audit"):
        s = s[: -len("_audit")]
    while True:
        m = _SUBSAMPLE_SUFFIX.search(s)
        if not m:
            break
        s = s[: m.start()]
    return s


def resolve_config(project: Path, log_path: Path, candidate: str) -> Path | None:
    """configs/v2/<candidate>.yaml for v2-tree logs; configs/v2_tdsc/tau/<candidate>.yaml then
    configs/v2_tdsc/<candidate>.yaml for v2_tdsc-tree logs (or any candidate starting 'tau_').
    Only ever looks in these documented locations -- never falls back to the top-level configs/
    dir (which holds a different, possibly-divergent generation of some of the same experiment
    ids) and never fuzzy-matches. If nothing is found here, the caller reports 'no config found'."""
    under_tdsc = "v2_tdsc" in log_path.parts
    if under_tdsc or candidate.startswith("tau_"):
        for rel in (f"configs/v2_tdsc/tau/{candidate}.yaml", f"configs/v2_tdsc/{candidate}.yaml"):
            p = project / rel
            if p.is_file():
                return p
        return None
    p = project / f"configs/v2/{candidate}.yaml"
    return p if p.is_file() else None


def read_model_base(config_path: Path) -> str | None | object:
    """Minimal stdlib line-scan for the 'model: \\n  base: "..."' block -- these configs are flat
    enough (no anchors, no multi-line scalars) that a real YAML parser is not needed and pyyaml is
    not an existing dependency of this project's scripts. Returns the raw base string, None for an
    explicit YAML null, or NO_MODEL_BLOCK if the config has no top-level 'model:' key at all (the
    v2_tdsc E5 a800 configs use a different alpha/beta schema)."""
    in_block = False
    for line in config_path.read_text().splitlines():
        if re.match(r"^model:\s*$", line):
            in_block = True
            continue
        if in_block:
            if re.match(r"^\S", line):  # dedent to a new top-level key: block over
                break
            m = re.match(r"^\s+base:\s*(.+?)\s*$", line)
            if m:
                raw = m.group(1)
                if raw in ("null", "~", ""):
                    return None
                return raw.strip('"').strip("'")
    return NO_MODEL_BLOCK if not in_block else None


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project", default=str(Path(__file__).resolve().parents[2]))
    args = ap.parse_args()
    project = Path(args.project)

    model_records: dict[str, int] = defaultdict(int)
    model_files: dict[str, set[str]] = defaultdict(set)
    model_check: dict[str, dict[str, int]] = defaultdict(lambda: {"ok": 0, "unavailable": 0})

    all_stats: list[FileStats] = []

    for label, rel, is_backup in GROUPS:
        d = project / rel
        files = sorted(d.glob("*.jsonl")) if d.is_dir() else []
        tag = "  [BACKUP/SECONDARY]" if is_backup else "  [PRIMARY]"
        print("\n" + "=" * 122)
        print(f"GROUP: {label}{tag}   ({d})   -- {len(files)} file(s)")
        print("=" * 122)
        if not files:
            print("  (no .jsonl files here)")
            continue
        hdr = f"{'file':<54}{'n':>6}  {'check_anomaly ok/unavail/total':<32}{'fn_size':>8}{'rec_size':>9}  {'rec_family':<12}  recorded model(s)"
        print(hdr)
        print("-" * len(hdr))
        for f in files:
            st = scan_file(f, label, is_backup, model_records, model_files, model_check)
            all_stats.append(st)
            ca = f"{st.check_anomaly_ok}/{st.check_anomaly_unavailable}/{st.check_anomaly_total}"
            fn_size = filename_implied_size(f.stem)
            if len(st.models) == 1:
                m = next(iter(st.models))
                rsize, rfam = recorded_size(m), recorded_family(m)
            elif len(st.models) > 1:
                rsize, rfam = "MIXED", "MIXED"
            else:
                rsize, rfam = "-", "-"
            extra = f" bad_json={st.bad_json}" if st.bad_json else ""
            print(f"{f.name:<54}{st.n_records:>6}  {ca:<32}{fn_size:>8}{rsize:>9}  {rfam:<12}  {fmt_models(st)}{extra}")

    total_records = sum(s.n_records for s in all_stats)
    print(f"\nTOTAL: {len(all_stats)} files scanned, {total_records} records, "
          f"{sum(1 for s in all_stats if s.is_backup)} of those files are backup/secondary.")

    # ------------------------------------------------------------------
    # task 4a: MISMATCH table -- filename says <N>B, recorded model says a different size.
    # A file with no size token in its name makes no claim, so it is excluded here (nothing to
    # contradict), not treated as a match either. Files with 0 recorded models (empty file / no
    # agent_config.model field) cannot be checked at all and are listed separately below instead
    # of being silently skipped or wrongly asserted as a match/mismatch.
    # ------------------------------------------------------------------
    print("\n" + "=" * 122)
    print("MISMATCH TABLE: filename-implied size != recorded size (task 4)")
    print("(files with no size token in the filename are excluded -- 'unspecified' makes no claim "
          "to contradict; see the unverifiable list further below for files with no recorded model at all)")
    print("=" * 122)
    mismatches = []
    unverifiable = []
    for st in all_stats:
        fn_size = filename_implied_size(st.path.stem)
        if not st.models:
            if fn_size != "unspecified":
                unverifiable.append((st, fn_size))
            continue
        rec_sizes = {recorded_size(m) for m in st.models}
        if fn_size == "unspecified":
            continue
        if fn_size not in rec_sizes or len(rec_sizes) > 1:
            mismatches.append((st, fn_size, rec_sizes))
    if mismatches:
        hdr = f"{'group':<28}{'file':<54}{'n':>6}{'fn_size':>9}{'rec_size':>10}  recorded model(s)"
        print(hdr)
        print("-" * len(hdr))
        for st, fn_size, rec_sizes in mismatches:
            rsize_str = "/".join(sorted(rec_sizes))
            print(f"{st.group:<28}{st.path.name:<54}{st.n_records:>6}{fn_size:>9}{rsize_str:>10}  {fmt_models(st)}")
    else:
        print("(none found)")
    print(f"\n{len(mismatches)} file(s) with a filename size token contradicted by the recorded model.")

    if unverifiable:
        print("\nUnverifiable (filename claims a size, but the file has no recorded model to check it "
              "against -- empty file or a different record schema with no agent_config.model field):")
        for st, fn_size in unverifiable:
            reason = "0 records (empty file)" if st.n_records == 0 else "no agent_config.model field in any record"
            print(f"  {st.group}/{st.path.name:<48} filename claims {fn_size}, {reason} (n={st.n_records}) -> UNRESOLVED")

    # ------------------------------------------------------------------
    # task 4b: MIXED table -- files with >1 distinct recorded model.
    # ------------------------------------------------------------------
    print("\n" + "=" * 122)
    print("MIXED-MODEL TABLE: files with more than one distinct recorded agent_config.model (task 4)")
    print("=" * 122)
    mixed = [st for st in all_stats if len(st.models) > 1]
    if mixed:
        for st in mixed:
            print(f"  {st.group}/{st.path.name}  n={st.n_records}  models={sorted(st.models)}")
    else:
        print(f"(none found -- every one of the {len(all_stats)} scanned files that has any recorded "
              f"model at all records exactly one distinct agent_config.model value)")

    # ------------------------------------------------------------------
    # task 5(a)
    # ------------------------------------------------------------------
    print("\n" + "=" * 122)
    print("QUESTION (a): any tool-channel log (E3_/E4_/tau_ filename, >=1 record with a non-empty")
    print("tool_chain containing a check_anomaly step) whose recorded model family is Llama-3.1 at 8B?")
    print("=" * 122)
    hits = []
    for st in all_stats:
        if not (st.path.name.startswith("E3_") or st.path.name.startswith("E4_") or st.path.name.startswith("tau_")):
            continue
        if not st.has_check_anomaly_step:
            continue
        if len(st.models) != 1:
            continue
        m = next(iter(st.models))
        if recorded_family(m) == "Llama-3.1" and recorded_size(m) == "8B":
            hits.append((st, m))
    print("YES" if hits else "NO")
    for st, m in hits:
        print(f"  {st.group}/{st.path.name}  n={st.n_records}  recorded_model={m}  "
              f"check_anomaly: ok={st.check_anomaly_ok} unavailable={st.check_anomaly_unavailable} "
              f"(total={st.check_anomaly_total})")
        if st.check_anomaly_ok == 0 and st.check_anomaly_total > 0:
            print(f"  CAVEAT: every check_anomaly step in this file is the harness's 'Unknown tool' "
                  f"guard -- the tool was requested but never actually available in this condition. "
                  f"By the literal definition above (step present, non-empty) this file counts; "
                  f"but there is no run anywhere in the corpus where check_anomaly *succeeded* with "
                  f"a Llama-3.1-8B model.")

    # ------------------------------------------------------------------
    # task 5(b)
    # ------------------------------------------------------------------
    print("\n" + "=" * 122)
    print("QUESTION (b): E3_noRF_8B / E3_noRF_14B_* / E3_noRF_32B_* -- what model actually ran?")
    print("=" * 122)
    for prefix in ("E3_noRF_8B_unsw_sub200_audit.jsonl", "E3_noRF_14B_", "E3_noRF_32B_"):
        matches = [st for st in all_stats if st.path.name.startswith(prefix)]
        if not matches:
            print(f"  {prefix}*  -> no matching file found")
            continue
        for st in matches:
            model_str = fmt_models(st)
            print(f"  {st.group}/{st.path.name}")
            print(f"      recorded model    : {model_str}")
            print(f"      tool_chain has check_anomaly steps : "
                  f"{'YES' if st.has_check_anomaly_step else 'NO'} "
                  f"(ok={st.check_anomaly_ok}, unavailable={st.check_anomaly_unavailable}, total={st.check_anomaly_total})")
            print(f"      record count      : {st.n_records}")

    # ------------------------------------------------------------------
    # task 5(c)
    # ------------------------------------------------------------------
    print("\n" + "=" * 122)
    print("QUESTION (c): every distinct recorded model string in the whole corpus (all 6 groups above)")
    print("=" * 122)
    hdr = f"{'recorded model':<44}{'files':>7}{'records':>10}  check_anomaly ever?"
    print(hdr)
    print("-" * len(hdr))
    for m in sorted(model_records):
        nf = len(model_files[m])
        nr = model_records[m]
        chk = model_check[m]
        if chk["ok"] > 0:
            ca = f"YES (ok={chk['ok']}, unavailable={chk['unavailable']})"
        elif chk["unavailable"] > 0:
            ca = f"ATTEMPTED ONLY -- {chk['unavailable']} steps, all 'Unknown tool' (never succeeded)"
        else:
            ca = "NO (no check_anomaly step recorded for this model anywhere)"
        print(f"{m:<44}{nf:>7}{nr:>10}  {ca}")

    # ------------------------------------------------------------------
    # task 6: config cross-check
    # ------------------------------------------------------------------
    print("\n" + "=" * 122)
    print("CONFIG CROSS-CHECK: model.base (YAML) vs recorded agent_config.model (task 6)")
    print("=" * 122)
    config_cache: dict[Path, object] = {}
    no_config: list[FileStats] = []
    null_base: list[FileStats] = []
    no_model_block: list[FileStats] = []
    agree: list[FileStats] = []
    disagree: list[tuple[FileStats, Path, str, set[str]]] = []
    found_but_unverifiable: list[tuple[FileStats, Path, str]] = []
    for st in all_stats:
        candidate = config_candidate_stem(st.path.stem)
        cfg = resolve_config(project, st.path, candidate)
        if cfg is None:
            no_config.append(st)
            continue
        if cfg not in config_cache:
            config_cache[cfg] = read_model_base(cfg)
        base = config_cache[cfg]
        if base is NO_MODEL_BLOCK:
            no_model_block.append(st)
            continue
        if base is None:
            null_base.append(st)
            continue
        if not st.models:
            # config resolved and has a real base, but the log has nothing recorded to compare it
            # against (empty file / no agent_config.model field) -- report the config's claim
            # rather than silently dropping it.
            found_but_unverifiable.append((st, cfg, base))
            continue
        norm_base = _API_PREFIX.sub("", base)
        if len(st.models) == 1 and norm_base in st.models:
            agree.append(st)
        else:
            disagree.append((st, cfg, base, st.models))

    print(f"\n{len(agree)} file(s): config model.base matches recorded agent_config.model exactly "
          f"(after stripping any 'api://host:port/' prefix).")

    if disagree:
        print(f"\n{len(disagree)} file(s) where config model.base DISAGREES with the recorded model:")
        for st, cfg, base, models in disagree:
            print(f"  {st.group}/{st.path.name}")
            print(f"      config   : {cfg.relative_to(project)}  ->  model.base = {base!r}")
            print(f"      recorded : {fmt_models(st)}")
    else:
        print("\n0 files where a resolved config's model.base disagrees with the recorded model.")

    if null_base:
        print(f"\n{len(null_base)} file(s) where the config declares model.base: null (no LLM configured "
              f"-- expected for the non-LLM RandomForest baseline, not counted as a disagreement):")
        for st in null_base:
            print(f"  {st.group}/{st.path.name}  recorded={fmt_models(st)}")

    if found_but_unverifiable:
        print(f"\n{len(found_but_unverifiable)} file(s) with a resolved config but nothing recorded "
              f"in the log to check it against (empty file / no agent_config.model field) -- "
              f"config's claim shown for reference, not cross-checked:")
        for st, cfg, base in found_but_unverifiable:
            print(f"  {st.group}/{st.path.name}  (n={st.n_records})")
            print(f"      config : {cfg.relative_to(project)}  ->  model.base = {base!r}  -> UNRESOLVED (no recorded model to compare)")

    if no_model_block:
        print(f"\n{len(no_model_block)} file(s) whose config has no top-level 'model:' key at all "
              f"(different experiment schema, e.g. the E5 alpha/beta consensus configs):")
        for st in no_model_block:
            print(f"  {st.group}/{st.path.name}")

    if no_config:
        print(f"\n{len(no_config)} file(s) with 'no config found' at the documented paths "
              f"(configs/v2/<id>.yaml, or configs/v2_tdsc/[tau/]<id>.yaml) -- not guessed:")
        for st in no_config:
            candidate = config_candidate_stem(st.path.stem)
            print(f"  {st.group}/{st.path.name}  (tried candidate id: {candidate!r})")


if __name__ == "__main__":
    main()
