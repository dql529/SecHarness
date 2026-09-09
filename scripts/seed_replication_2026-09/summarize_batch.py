#!/usr/bin/env python3
"""summarize_batch.py — registry-style summary of one multi-seed batch.

For every *_seed<S>_<TAG>_results.json in RESULTS_DIR the row shows acc / FPR / ZD / escalation rate /
escalation error lift (P(error|escalated)/P(error|not escalated), from the audit log) / chain / ms per sample /
sha256[:16] of the audit log. Then per-condition mean ± sample std (ddof=1), and — when an archived log is given
for a condition — that run's metrics and the number of identical per-sample binary verdicts.

Usage: summarize_batch.py <TAG> <results_dir> <logs_dir> [cond=<exp_id>[:<archived_audit_log>] ...]
  e.g. summarize_batch.py 20260909T1932 project/results/tables/v2_tdsc project/logs/v2_tdsc \
        "14B noPermissions=tau_13B_noPermissions_unsw:project/logs/v2_tdsc/tau_13B_noPermissions_unsw_sub200_audit.jsonl"
"""
import glob, hashlib, json, os, re, statistics, sys

def load_log(p):
    recs = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    return {r["sample_index"]: r for r in recs}

def log_metrics(recs):
    rs = list(recs.values()); n = len(rs)
    err = lambda xs: (sum(1 for r in xs if not r["evaluation"]["detection_correct"]) / len(xs)) if xs else float("nan")
    esc = [r for r in rs if r["result"].get("escalated")]; non = [r for r in rs if not r["result"].get("escalated")]
    e_esc, e_non = err(esc), err(non)
    lift = (e_esc / e_non) if (esc and non and e_non > 0) else float("nan")
    correct = sum(1 for r in rs if r["evaluation"]["detection_correct"])
    fp = sum(1 for r in rs if r["evaluation"]["binary_gt"] == "benign" and r["evaluation"]["binary_pred"] != "benign")
    neg = sum(1 for r in rs if r["evaluation"]["binary_gt"] == "benign")
    zd = [r for r in rs if r["input"].get("is_zeroday")]
    zdr = (sum(1 for r in zd if r["evaluation"]["detection_correct"]) / len(zd)) if zd else float("nan")
    chain = statistics.mean(r["result"].get("total_steps", 0) for r in rs) if rs else float("nan")
    ms = statistics.mean(r["efficiency"]["total_latency_ms"] for r in rs) if rs else float("nan")
    return dict(n=n, acc=correct / n, fpr=fp / neg if neg else float("nan"), zd=zdr, esc=len(esc) / n, lift=lift, chain=chain, ms=ms)

def sha16(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""): h.update(chunk)
    return h.hexdigest()[:16]

def fmt(x, d=3):
    return "nan" if x != x else f"{x:.{d}f}"

def main():
    tag, rdir, ldir = sys.argv[1:4]
    conds = []
    for spec in sys.argv[4:]:
        label, rest = spec.split("=", 1)
        exp_id, _, ref = rest.partition(":")
        conds.append((label, exp_id, ref or None))
    print(f"| cond | seed | acc | FPR | ZD | esc. rate | esc. error lift | chain | ms/sample | audit log sha256[:16] |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    stats = {}
    for label, exp_id, ref in conds:
        rows = []
        pat = re.compile(rf"^{re.escape(exp_id)}_seed(\d+)_{re.escape(tag)}_results\.json$")
        cands = [(int(m.group(1)), rj) for rj in glob.glob(os.path.join(rdir, f"{exp_id}_seed*_{tag}_results.json"))
                 if (m := pat.match(os.path.basename(rj)))]  # exact form only: a pilot file (…_seed42_pilot_TAG…) is excluded
        for seed, rj in sorted(cands):
            r = json.load(open(rj))
            logp = [q for q in glob.glob(os.path.join(ldir, f"{exp_id}_*seed{seed}_{tag}_audit.jsonl"))
                    if re.match(rf"^{re.escape(exp_id)}_(sub\d+|pilot\d+)?_?seed{seed}_{re.escape(tag)}_audit\.jsonl$", os.path.basename(q))]
            if len(logp) != 1: print(f"!! {exp_id} seed {seed}: {len(logp)} audit logs found"); continue
            recs = load_log(logp[0]); lm = log_metrics(recs)
            if lm["n"] != r["n_samples"]: print(f"!! {exp_id} seed {seed}: log n={lm['n']} vs results n={r['n_samples']}")
            for k, a, b in (("acc", r["accuracy"], lm["acc"]), ("fpr", r["fpr"], lm["fpr"]), ("esc", r["escalation_rate"], lm["esc"])):
                if abs(a - b) > 5e-4: print(f"!! {exp_id} seed {seed}: results.json {k}={a} vs audit-derived {b}")
            row = dict(seed=seed, acc=r["accuracy"], fpr=r["fpr"], zd=r["zeroday_detection_rate"], esc=r["escalation_rate"], lift=lm["lift"], chain=r["avg_chain_length"], ms=r["avg_latency_ms"], sha=sha16(logp[0]), recs=recs)
            rows.append(row)
            print(f"| {label} | {seed} | {fmt(row['acc'])} | {fmt(row['fpr'])} | {fmt(row['zd'],2)} | {fmt(row['esc'])} | {fmt(row['lift'],2)} | {fmt(row['chain'],2)} | {row['ms']:.0f} | `{row['sha']}` |")
        stats[label] = (rows, ref)
    print()
    for label, (rows, ref) in stats.items():
        if not rows: print(f"{label}: no runs found for tag {tag}"); continue
        def ms(k, d=3):
            xs = [r[k] for r in rows if r[k] == r[k]]
            if not xs: return "nan"
            sd = statistics.stdev(xs) if len(xs) > 1 else 0.0
            return f"{statistics.mean(xs):.{d}f} ± {sd:.{d}f}"
        print(f"Seed statistics {label} (n={len(rows)}): acc {ms('acc')}, FPR {ms('fpr')}, ZD {ms('zd')}, escalation {ms('esc')}, escalation error lift {ms('lift')}, chain {ms('chain',2)}")
        if ref and os.path.exists(ref):
            A = load_log(ref); am = log_metrics(A)
            print(f"  archived {os.path.basename(ref)}: n={am['n']} acc {fmt(am['acc'])}, FPR {fmt(am['fpr'])}, ZD {fmt(am['zd'],2)}, escalation {fmt(am['esc'])}, lift {fmt(am['lift'],2)}, chain {fmt(am['chain'],2)}")
            for r in rows:
                common = sorted(set(A) & set(r["recs"]))
                same = sum(1 for i in common if A[i]["evaluation"]["binary_pred"] == r["recs"][i]["evaluation"]["binary_pred"])
                print(f"    seed {r['seed']}: identical binary verdicts {same}/{len(common)}")
        elif ref:
            print(f"  archived log {ref} NOT FOUND")

if __name__ == "__main__":
    main()
