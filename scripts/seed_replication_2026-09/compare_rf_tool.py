#!/usr/bin/env python3
"""compare_rf_tool.py — is the random-forest tool identical between two audit logs?

For a replication run to be comparable with the archived batch, the deterministic part
(the 200-sample subset and the check_anomaly verdict+confidence per sample) must match
exactly; only the LLM's behaviour may differ. This script joins two AuditRecordV2 JSONL
logs on sample_index, checks that the input text is byte-identical, and compares the
check_anomaly tool result (class, confidence) per sample.

Usage: compare_rf_tool.py <log_a.jsonl> <log_b.jsonl>
Exit 0 = same subset and identical RF results; 2 = differences (listed); 3 = no overlap.
"""
import json, sys, hashlib

def load(path):
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            out[r["sample_index"]] = r
    return out

def rf_result(r):
    for tc in r.get("tool_chain", []):
        if tc.get("tool") == "check_anomaly":
            res = tc.get("result") or {}
            if isinstance(res, str):
                try: res = json.loads(res)
                except Exception: return ("unparsed", res[:60])
            return (res.get("class") or res.get("prediction") or res.get("verdict"),
                    round(float(res.get("confidence", -1)), 6) if res.get("confidence") is not None else None)
    return None

def input_hash(r):
    inp = r.get("input")
    s = json.dumps(inp, sort_keys=True, ensure_ascii=False) if not isinstance(inp, str) else inp
    return hashlib.sha256(s.encode()).hexdigest()[:16]

def main(a, b):
    A, B = load(a), load(b)
    common = sorted(set(A) & set(B))
    print(f"records: A={len(A)} B={len(B)} common sample_index={len(common)}")
    if not common:
        print("NO OVERLAP"); return 3
    input_diff = [i for i in common if input_hash(A[i]) != input_hash(B[i])]
    rf_a_missing = [i for i in common if rf_result(A[i]) is None]
    rf_b_missing = [i for i in common if rf_result(B[i]) is None]
    rf_diff = [i for i in common if rf_result(A[i]) is not None and rf_result(B[i]) is not None and rf_result(A[i]) != rf_result(B[i])]
    print(f"input text differs: {len(input_diff)}  | check_anomaly absent: A={len(rf_a_missing)} B={len(rf_b_missing)}  | RF result differs: {len(rf_diff)}")
    for i in (input_diff[:5]): print(f"  input differs @{i}")
    for i in (rf_diff[:10]): print(f"  RF differs @{i}: A={rf_result(A[i])} B={rf_result(B[i])}")
    ok = not input_diff and not rf_diff and len(common) >= 0.9 * min(len(A), len(B))
    print("VERDICT:", "SAME_SUBSET_SAME_RF" if ok else "DIFFERENT")
    return 0 if ok else 2

if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
