#!/usr/bin/env python3
"""
run_tau_orchestrator.py — Disk-managed orchestrator for Exp-τ ablation conditions.

Manages vLLM server lifecycle, disk budget, and sequential execution of all
11 new tau ablation conditions across 7B (full-precision), 13B-AWQ, and 32B-AWQ.

CONTRACT: main() -> None
    1. Verify environment (vllm, transformers importable; HF_ENDPOINT set; screen available)
    2. Disk pre-check: abort if <10GB free
    3. 7B-noTools (single condition, full-precision HF direct):
       a. Download Qwen/Qwen2.5-7B-Instruct if missing (~14GB)
       b. run_experiment.py --config tau_7B_noTools_unsw.yaml
       c. If --model-delete-after-use: rm -rf the 7B cache dir
    4. For each AWQ model_size in [14B, 32B]:
       a. Disk check -> abort if <10GB free
       b. Download AWQ model if missing (huggingface-cli with HF_ENDPOINT=hf-mirror.com)
       c. Start vLLM server in screen 'vllm_tau'; readiness probe (curl /v1/models, 300s timeout)
       d. For each ablation config of this size: run experiment, append row to tau_results.csv
       e. Stop vLLM server: fuser -k 8000/tcp; screen -S vllm_tau -X quit
       f. If --model-delete-after-use: rm -rf this model's hub dir
    5. Emit summary: conditions attempted, N success / N fail

CLI: python run_tau_orchestrator.py [--dry-run] [--skip-sizes 32B] [--no-delete-after-use] [--resume]
Default: --model-delete-after-use ENABLED.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger("tau-orchestrator")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PYTHON = "/root/miniconda3/bin/python3"
HF_CLI = "/root/miniconda3/bin/huggingface-cli"
SCRATCH_DIR = Path(os.environ.get("SCRATCH_DIR", "/root/autodl-tmp"))
HF_HOME = Path(os.environ.get("HF_HOME", str(SCRATCH_DIR / "huggingface")))
HF_HUB_DIR = HF_HOME / "hub"

# Project root: two levels up from this script (project/scripts/v2_tdsc/)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUN_EXPERIMENT = PROJECT_ROOT / "scripts" / "v2" / "run_experiment.py"
CONFIG_DIR = PROJECT_ROOT / "configs" / "v2_tdsc" / "tau"
RESULTS_CSV = PROJECT_ROOT / "results" / "tables" / "v2_tdsc" / "tau_results.csv"
LOG_DIR = PROJECT_ROOT / "logs" / "v2_tdsc"

DISK_MIN_GB = 10.0
VLLM_PORT = 8000
VLLM_SCREEN = "vllm_tau"
VLLM_READINESS_TIMEOUT_S = 300

# Model repos and expected HF hub directory names
MODEL_SPECS: dict[str, dict[str, str]] = {
    "7B": {
        "repo": "Qwen/Qwen2.5-7B-Instruct",
        "hub_dir": "models--Qwen--Qwen2.5-7B-Instruct",
        "quantization": None,
        "vllm_args": "",
    },
    "13B": {
        "repo": "Qwen/Qwen2.5-14B-Instruct-AWQ",
        "hub_dir": "models--Qwen--Qwen2.5-14B-Instruct-AWQ",
        "quantization": "awq_marlin",
        "vllm_args": "--quantization awq_marlin --dtype auto",
    },
    "32B": {
        "repo": "Qwen/Qwen2.5-32B-Instruct-AWQ",
        "hub_dir": "models--Qwen--Qwen2.5-32B-Instruct-AWQ",
        "quantization": "awq_marlin",
        "vllm_args": "--quantization awq_marlin --dtype auto",
    },
}

# Ablation conditions per model size
CONDITIONS: dict[str, list[str]] = {
    "7B": ["noTools"],
    "13B": ["full", "noKnowledge", "noObservation", "noPermissions", "noTools"],
    "32B": ["full", "noKnowledge", "noObservation", "noPermissions", "noTools"],
}

CSV_FIELDNAMES = [
    "model_size", "ablation", "config", "status",
    "accuracy", "fpr", "f1", "n_samples",
    "started_at", "finished_at", "error_msg",
]


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def get_free_disk_gb(path: Path) -> float:
    """Return free disk space in GB for the filesystem containing path."""
    stat = shutil.disk_usage(path)
    return stat.free / (1024 ** 3)


def check_disk(path: Path, min_gb: float, dry_run: bool = False) -> None:
    """Abort if free disk < min_gb GB."""
    free = get_free_disk_gb(path)
    log.info("Disk free: %.1f GB (required: %.1f GB)", free, min_gb)
    if free < min_gb:
        msg = f"Disk exhausted: {free:.1f} GB free, need {min_gb} GB. Aborting."
        log.error(msg)
        raise RuntimeError(msg)
    if dry_run:
        log.info("[dry-run] Disk check passed (%.1f GB free)", free)


def verify_environment() -> None:
    """Check required tools and environment variables are available."""
    log.info("Verifying environment...")
    # Check HF_ENDPOINT
    hf_endpoint = os.environ.get("HF_ENDPOINT", "")
    if not hf_endpoint:
        log.warning("HF_ENDPOINT not set; downloads may fail in China region. "
                    "Set: export HF_ENDPOINT=https://hf-mirror.com")
    else:
        log.info("HF_ENDPOINT=%s", hf_endpoint)

    # Check screen
    if shutil.which("screen") is None:
        raise EnvironmentError("screen is not available in PATH. Install with: apt-get install screen")

    # Check fuser
    if shutil.which("fuser") is None:
        log.warning("fuser not found; zombie cleanup via fuser -k will be skipped (use kill manually)")

    # Check Python + vllm importable
    result = subprocess.run(
        [PYTHON, "-c", "import vllm, transformers; print(vllm.__version__, transformers.__version__)"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise EnvironmentError(
            f"Python at {PYTHON} cannot import vllm/transformers:\n{result.stderr}"
        )
    log.info("vllm + transformers OK: %s", result.stdout.strip())

    # Check run_experiment.py exists
    if not RUN_EXPERIMENT.exists():
        raise FileNotFoundError(f"run_experiment.py not found at {RUN_EXPERIMENT}")
    log.info("run_experiment.py found: %s", RUN_EXPERIMENT)

    log.info("Environment verification passed.")


def load_results_csv() -> dict[tuple[str, str], dict[str, str]]:
    """Load existing tau_results.csv into {(model_size, ablation): row} dict."""
    if not RESULTS_CSV.exists():
        return {}
    results: dict[tuple[str, str], dict[str, str]] = {}
    with open(RESULTS_CSV, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["model_size"], row["ablation"])
            results[key] = row
    log.info("Loaded %d existing rows from %s", len(results), RESULTS_CSV)
    return results


def append_results_csv(row: dict[str, str]) -> None:
    """Append a single row to tau_results.csv, creating file+header if needed."""
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    write_header = not RESULTS_CSV.exists()
    with open(RESULTS_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def download_model(repo: str, dry_run: bool = False) -> None:
    """Download HF model using huggingface-cli with HF mirror."""
    log.info("Downloading model: %s", repo)
    env = os.environ.copy()
    env.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    cmd = [HF_CLI, "download", repo]
    if dry_run:
        log.info("[dry-run] Would run: %s", " ".join(cmd))
        return
    result = subprocess.run(cmd, env=env, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"huggingface-cli download failed for {repo} (exit {result.returncode})")
    log.info("Download complete: %s", repo)


def model_cached(hub_dir_name: str) -> bool:
    """Check if model hub directory exists and has snapshots."""
    model_dir = HF_HUB_DIR / hub_dir_name
    if not model_dir.exists():
        return False
    snapshots = model_dir / "snapshots"
    return snapshots.exists() and any(snapshots.iterdir())


def delete_model_cache(hub_dir_name: str, dry_run: bool = False) -> None:
    """Delete model hub directory to reclaim disk space."""
    model_dir = HF_HUB_DIR / hub_dir_name
    if not model_dir.exists():
        log.info("Cache dir not found (already deleted?): %s", model_dir)
        return
    if dry_run:
        log.info("[dry-run] Would delete cache: %s", model_dir)
        return
    log.info("Deleting model cache: %s", model_dir)
    shutil.rmtree(model_dir)
    log.info("Deleted: %s", model_dir)


def kill_port(port: int, dry_run: bool = False) -> None:
    """Kill any process listening on the given TCP port (zombie cleanup)."""
    if shutil.which("fuser") is None:
        log.warning("fuser not available; skipping port kill for %d", port)
        return
    cmd = ["fuser", "-k", f"{port}/tcp"]
    if dry_run:
        log.info("[dry-run] Would run: %s", " ".join(cmd))
        return
    # fuser returns 1 if no process found — that's OK
    subprocess.run(cmd, capture_output=True)
    log.info("fuser -k %d/tcp executed (zombie cleanup)", port)


def start_vllm_server(repo: str, vllm_args: str, dry_run: bool = False) -> None:
    """Start vLLM server in a detached screen session."""
    log_file = LOG_DIR / "vllm_tau.log"
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    cmd = (
        f"export PATH=/root/miniconda3/bin:$PATH; "
        f"export HF_HOME={HF_HOME}; "
        f"{PYTHON} -m vllm.entrypoints.openai.api_server "
        f"--model {repo} "
        f"--port {VLLM_PORT} "
        f"--gpu-memory-utilization 0.85 "
        f"{vllm_args} "
        f"2>&1 | tee {log_file}"
    )
    screen_cmd = ["screen", "-dmS", VLLM_SCREEN, "bash", "-c", cmd]

    if dry_run:
        log.info("[dry-run] Would start vLLM server: %s", " ".join(screen_cmd))
        return

    log.info("Starting vLLM server for %s in screen '%s'...", repo, VLLM_SCREEN)
    result = subprocess.run(screen_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to start vLLM screen session:\n{result.stderr}"
        )
    log.info("vLLM server screen '%s' started.", VLLM_SCREEN)


def wait_vllm_ready(timeout_s: int = VLLM_READINESS_TIMEOUT_S, dry_run: bool = False) -> bool:
    """
    Poll /v1/models until server responds or timeout.
    Returns True if server became ready, False on timeout.
    """
    if dry_run:
        log.info("[dry-run] Skipping vLLM readiness probe.")
        return True

    url = f"http://localhost:{VLLM_PORT}/v1/models"
    deadline = time.time() + timeout_s
    log.info("Waiting for vLLM server at %s (timeout=%ds)...", url, timeout_s)
    while time.time() < deadline:
        result = subprocess.run(
            ["curl", "-sf", url],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            log.info("vLLM server is ready.")
            return True
        time.sleep(5)
    log.error("vLLM server did not become ready within %ds.", timeout_s)
    return False


def stop_vllm_server(dry_run: bool = False) -> None:
    """Stop vLLM server: kill port then quit screen session."""
    kill_port(VLLM_PORT, dry_run=dry_run)
    if dry_run:
        log.info("[dry-run] Would quit screen: screen -S %s -X quit", VLLM_SCREEN)
        return
    subprocess.run(["screen", "-S", VLLM_SCREEN, "-X", "quit"], capture_output=True)
    log.info("vLLM server stopped (screen '%s' quit).", VLLM_SCREEN)


def run_single_experiment(
    model_size: str,
    ablation: str,
    dry_run: bool = False,
) -> dict[str, str]:
    """
    CONTRACT: run_single_experiment(model_size, ablation, dry_run) -> dict
        model_size: str  — "7B", "13B", or "32B"
        ablation: str    — e.g. "full", "noKnowledge", etc.
        dry_run: bool    — if True, skip actual execution
        returns: CSV row dict with status, metrics, timestamps
    """
    config_name = f"tau_{model_size}_{ablation}_unsw.yaml"
    config_path = CONFIG_DIR / config_name
    started_at = datetime.now(timezone.utc).isoformat()
    row: dict[str, str] = {
        "model_size": model_size,
        "ablation": ablation,
        "config": config_name,
        "status": "fail",
        "accuracy": "",
        "fpr": "",
        "f1": "",
        "n_samples": "",
        "started_at": started_at,
        "finished_at": "",
        "error_msg": "",
    }

    if not config_path.exists():
        row["error_msg"] = f"Config not found: {config_path}"
        log.error("Config not found: %s", config_path)
        return row

    cmd = [PYTHON, str(RUN_EXPERIMENT), "--config", str(config_path)]
    log.info("Running: %s", " ".join(cmd))

    if dry_run:
        log.info("[dry-run] Would run experiment: %s/%s", model_size, ablation)
        row["status"] = "dry-run"
        row["finished_at"] = datetime.now(timezone.utc).isoformat()
        return row

    result = subprocess.run(cmd, capture_output=True, text=True)
    row["finished_at"] = datetime.now(timezone.utc).isoformat()

    if result.returncode != 0:
        row["error_msg"] = result.stderr[-500:] if result.stderr else "unknown error"
        log.error("Experiment failed: %s/%s\n%s", model_size, ablation, row["error_msg"])
        return row

    # Try to parse metrics from JSON results file
    exp_id = f"tau_{model_size}_{ablation}_unsw"
    results_json = PROJECT_ROOT / "results" / "tables" / "v2_tdsc" / f"{exp_id}_results.json"
    if results_json.exists():
        try:
            with open(results_json) as f:
                metrics = json.load(f)
            row["accuracy"] = str(metrics.get("accuracy", ""))
            row["fpr"] = str(metrics.get("fpr", ""))
            row["f1"] = str(metrics.get("f1", ""))
            row["n_samples"] = str(metrics.get("n_samples", ""))
            row["status"] = "success"
            log.info(
                "Experiment succeeded: %s/%s  acc=%.4f fpr=%.4f f1=%.4f",
                model_size, ablation,
                metrics.get("accuracy", 0),
                metrics.get("fpr", 0),
                metrics.get("f1", 0),
            )
        except (json.JSONDecodeError, KeyError) as exc:
            row["error_msg"] = f"JSON parse error: {exc}"
            log.warning("Could not parse results JSON: %s", exc)
            row["status"] = "success_no_metrics"
    else:
        log.warning("Results JSON not found at %s; marking success without metrics", results_json)
        row["status"] = "success_no_metrics"

    return row


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def main() -> None:
    """
    CONTRACT: main() -> None
        Orchestrates all 11 tau ablation conditions with disk/server lifecycle management.
        See module docstring for full contract.
    """
    ap = argparse.ArgumentParser(
        description="Tau ablation orchestrator for SecHarness Exp-τ (T3 validation)"
    )
    ap.add_argument("--dry-run", action="store_true",
                    help="Print commands without executing; no server/download/experiment")
    ap.add_argument("--skip-sizes", nargs="*", default=[],
                    metavar="SIZE", help="Skip model sizes, e.g. --skip-sizes 32B")
    ap.add_argument("--no-delete-after-use", action="store_true",
                    help="Disable model cache deletion after use (default: delete to save disk)")
    ap.add_argument("--resume", action="store_true",
                    help="Skip conditions already present in tau_results.csv with status=success")
    args = ap.parse_args()

    delete_after_use = not args.no_delete_after_use
    dry_run = args.dry_run
    skip_sizes = set(args.skip_sizes or [])

    log.info(
        "=== Tau Orchestrator === dry_run=%s  delete_after_use=%s  skip_sizes=%s  resume=%s",
        dry_run, delete_after_use, skip_sizes, args.resume,
    )

    # Step 1: Environment verification
    if not dry_run:
        verify_environment()
    else:
        log.info("[dry-run] Skipping environment verification.")

    # Step 2: Initial disk check
    _disk_check_path = HF_HOME if HF_HOME.exists() else (
        SCRATCH_DIR if SCRATCH_DIR.exists() else Path.home()
    )
    check_disk(_disk_check_path, DISK_MIN_GB, dry_run=dry_run)

    # Load existing results for resume
    existing_results = load_results_csv() if args.resume else {}

    n_attempted = 0
    n_success = 0
    n_fail = 0
    n_skipped = 0

    # Step 3: 7B-noTools (full-precision, no vLLM server needed)
    if "7B" not in skip_sizes:
        spec_7b = MODEL_SPECS["7B"]
        log.info("--- Model: 7B (full-precision) ---")

        resume_key = ("7B", "noTools")
        if args.resume and existing_results.get(resume_key, {}).get("status") == "success":
            log.info("skip 7B/noTools [already succeeded in tau_results.csv]")
            n_skipped += 1
        else:
            # Disk check before download
            check_disk(
                HF_HOME if HF_HOME.exists() else (
                    SCRATCH_DIR if SCRATCH_DIR.exists() else Path.home()
                ),
                DISK_MIN_GB, dry_run=dry_run,
            )

            # Download if missing
            if dry_run or not model_cached(spec_7b["hub_dir"]):
                download_model(spec_7b["repo"], dry_run=dry_run)
            else:
                log.info("7B model already cached, skipping download.")

            # Run experiment
            n_attempted += 1
            row = run_single_experiment("7B", "noTools", dry_run=dry_run)
            append_results_csv(row)
            if row["status"] in ("success", "success_no_metrics", "dry-run"):
                n_success += 1
                log.info("7B/noTools completed with status=%s", row["status"])
            else:
                n_fail += 1
                log.error("7B/noTools FAILED: %s", row["error_msg"])

            # Delete cache if enabled
            if delete_after_use:
                delete_model_cache(spec_7b["hub_dir"], dry_run=dry_run)
    else:
        log.info("Skipping 7B (in --skip-sizes).")

    # Step 4: AWQ models (13B, 32B) — require vLLM server
    for size in ["13B", "32B"]:
        if size in skip_sizes:
            log.info("Skipping %s (in --skip-sizes).", size)
            continue

        spec = MODEL_SPECS[size]
        ablations = CONDITIONS[size]
        log.info("--- Model: %s (%s) | Ablations: %s ---", size, spec["repo"], ablations)

        # Disk check before download
        try:
            check_disk(
                HF_HOME if HF_HOME.exists() else (
                    SCRATCH_DIR if SCRATCH_DIR.exists() else Path.home()
                ),
                DISK_MIN_GB, dry_run=dry_run,
            )
        except RuntimeError as exc:
            log.error("Disk check failed before %s download: %s", size, exc)
            for abl in ablations:
                n_attempted += 1
                n_fail += 1
                append_results_csv({
                    "model_size": size, "ablation": abl, "config": f"tau_{size}_{abl}_unsw.yaml",
                    "status": "fail", "accuracy": "", "fpr": "", "f1": "", "n_samples": "",
                    "started_at": datetime.now(timezone.utc).isoformat(), "finished_at": datetime.now(timezone.utc).isoformat(),
                    "error_msg": str(exc),
                })
            continue

        # Download AWQ model if missing
        if dry_run or not model_cached(spec["hub_dir"]):
            download_model(spec["repo"], dry_run=dry_run)
        else:
            log.info("%s model already cached, skipping download.", size)

        # Kill any zombie on port before starting
        kill_port(VLLM_PORT, dry_run=dry_run)

        # Start vLLM server
        try:
            start_vllm_server(spec["repo"], spec["vllm_args"], dry_run=dry_run)
        except RuntimeError as exc:
            log.error("Failed to start vLLM server for %s: %s", size, exc)
            for abl in ablations:
                n_attempted += 1
                n_fail += 1
                append_results_csv({
                    "model_size": size, "ablation": abl, "config": f"tau_{size}_{abl}_unsw.yaml",
                    "status": "fail", "accuracy": "", "fpr": "", "f1": "", "n_samples": "",
                    "started_at": datetime.now(timezone.utc).isoformat(), "finished_at": datetime.now(timezone.utc).isoformat(),
                    "error_msg": f"vLLM start failed: {exc}",
                })
            continue

        # Readiness probe
        server_ready = wait_vllm_ready(dry_run=dry_run)
        if not server_ready:
            log.error("vLLM server for %s did not become ready; skipping all ablations.", size)
            stop_vllm_server(dry_run=dry_run)
            for abl in ablations:
                n_attempted += 1
                n_fail += 1
                append_results_csv({
                    "model_size": size, "ablation": abl, "config": f"tau_{size}_{abl}_unsw.yaml",
                    "status": "fail", "accuracy": "", "fpr": "", "f1": "", "n_samples": "",
                    "started_at": datetime.now(timezone.utc).isoformat(), "finished_at": datetime.now(timezone.utc).isoformat(),
                    "error_msg": "vLLM readiness timeout",
                })
            if delete_after_use:
                delete_model_cache(spec["hub_dir"], dry_run=dry_run)
            continue

        # Run ablations
        for abl in ablations:
            resume_key = (size, abl)
            if args.resume and existing_results.get(resume_key, {}).get("status") == "success":
                log.info("skip %s/%s [already succeeded in tau_results.csv]", size, abl)
                n_skipped += 1
                continue

            n_attempted += 1
            row = run_single_experiment(size, abl, dry_run=dry_run)
            append_results_csv(row)
            if row["status"] in ("success", "success_no_metrics", "dry-run"):
                n_success += 1
                log.info("%s/%s completed with status=%s", size, abl, row["status"])
            else:
                n_fail += 1
                log.error("%s/%s FAILED: %s", size, abl, row["error_msg"])

        # Stop vLLM server after all ablations for this size
        stop_vllm_server(dry_run=dry_run)

        # Delete model cache if enabled
        if delete_after_use:
            delete_model_cache(spec["hub_dir"], dry_run=dry_run)

    # Step 5: Summary
    log.info(
        "=== Orchestration Complete === attempted=%d  success=%d  fail=%d  skipped=%d",
        n_attempted, n_success, n_fail, n_skipped,
    )
    log.info("Results written to: %s", RESULTS_CSV)
    if n_fail > 0:
        log.warning("%d condition(s) failed. Check tau_results.csv for error_msg.", n_fail)
        sys.exit(1)


if __name__ == "__main__":
    main()
