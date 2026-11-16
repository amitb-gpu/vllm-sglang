"""Gate and measure vLLM LMCache reloads with a matched recompute control."""

import argparse
import json
import re
import statistics
import subprocess
import time
from dataclasses import asdict

from bench_phase2 import URL, clean, document, load_tokenizer, metric_delta, metrics, save, server, snapshot_file, utc_now, with_context
from benchmark import WARMUP_TICKET, p95
from triage import TriageError, triage

TARGETS = (1024, 2048, 4096, 8192)
CONTEXT = 8704
WARMUPS = 2
TRIALS = 10
CHUNK = 256
STORE = re.compile(r"Stored (\d+) out of total (\d+) tokens")
RELOAD = re.compile(r"Retrieved (\d+) out of (\d+) required tokens \(from (\d+) total tokens\)")
BACKENDS = re.compile(r"LocalCPUBackend|LocalDiskBackend")
TICKET = "I was billed twice for the same monthly subscription. Please refund the duplicate charge."


def read_since(log, offset):
    with log.open(encoding="utf-8", errors="replace") as file:
        file.seek(offset)
        return file.read()


def wait_for(read, timeout=12):
    until = time.monotonic() + timeout
    while True:
        value = read()
        if value or time.monotonic() >= until:
            return value
        time.sleep(0.05)


def disk_files(path):
    return {file.name: file.stat().st_size for file in path.rglob("*") if file.is_file()}


def trial(mode, document_text, target, log, disk_path, bytes_per_token):
    result = {}
    try:
        if mode != "cold":
            files_before = disk_files(disk_path) if mode == "disk" else {}
            offset = log.stat().st_size
            triage(with_context(WARMUP_TICKET, document_text), URL)
            def stored_chunks():
                events = STORE.findall(read_since(log, offset))
                return events if sum(int(event[0]) for event in events) >= target else None

            stored = wait_for(stored_chunks)
            result["store_events"] = stored
            if not stored:
                raise TriageError("LMCache did not confirm storing the shared prefix")
            if mode == "disk":
                expected = sum(int(event[0]) // CHUNK for event in stored)

                def persisted():
                    files = disk_files(disk_path)
                    added = [size for name, size in files.items() if name not in files_before and size > 0]
                    return added if len(added) >= expected and sum(added) >= target * bytes_per_token else None

                sizes = wait_for(persisted)
                if not sizes:
                    raise TriageError("LMCache store did not produce the expected disk chunks")
                result.update(disk_files_added=len(sizes), disk_bytes_added=sum(sizes))
        offset = log.stat().st_size
        before = metrics()
        result.update(asdict(triage(with_context(TICKET, document_text), URL)))
        after = metrics()
        result["metrics_delta"] = metric_delta(before, after)
        if mode != "cold":
            hits = wait_for(lambda: RELOAD.findall(read_since(log, offset)))
            result["retrieved_tokens"] = sum(int(event[0]) for event in hits)
            result["retrieve_events"] = hits
            external_hits = sum(value for key, value in result["metrics_delta"].items()
                                if key.startswith("vllm:external_prefix_cache_hits_total"))
            if result["retrieved_tokens"] < target or external_hits < target:
                raise TriageError("Connector log or external-hit counter did not confirm the shared prefix")
    except TriageError as error:
        result["error"] = clean(str(error))
    return result


def measure(mode, gate=False):
    tokenizer = load_tokenizer()
    config = json.loads(snapshot_file("config.json").read_text(encoding="utf-8"))
    # K and V, each BF16, across all KV heads and transformer layers.
    bytes_per_token = 4 * config["num_hidden_layers"] * config["num_key_value_heads"] * config["head_dim"]
    report = {"mode": mode, "gate": gate, "started_at_utc": utc_now(), "cells": []}
    with server("vllm", "off", offload="none" if mode == "cold" else mode,
                context=CONTEXT, lmcache_env=True) as (metadata, log):
        report["server"] = metadata
        disk_path = log.parent / "kv-disk"
        if mode == "disk":
            report["disk_filesystem"] = subprocess.check_output(
                ["findmnt", "--target", str(disk_path), "-no", "FSTYPE"], text=True).strip()
            if report["disk_filesystem"] == "tmpfs":
                raise RuntimeError("Local disk trial requires a disk-backed directory")
        report["backend_startup_log"] = [
            clean(line) for line in log.read_text(errors="replace").splitlines()
            if BACKENDS.search(line)
        ]
        for target in TARGETS[:1] if gate else TARGETS:
            runs = []
            for index in range(1 if gate else WARMUPS + TRIALS):
                text = document(tokenizer, target, f"{target}-{index:02d}")
                run = trial(mode, text, target, log, disk_path, bytes_per_token)
                run.update(trial=index, warmup=not gate and index < WARMUPS,
                           document_tokens=len(tokenizer.encode(text, add_special_tokens=False).ids))
                runs.append(run)
                if gate and "error" in run:
                    break
            timings = [run["ttft_ms"] for run in runs if "error" not in run and not run["warmup"]]
            cell = {"target_tokens": target, "outcomes": runs, "successful": len(timings),
                    "errors": sum("error" in run and not run["warmup"] for run in runs),
                    "median_ttft_ms": round(statistics.median(timings), 2) if timings else None,
                    "p95_ttft_ms": p95(timings)}
            report["cells"].append(cell)
            print(mode, target, cell["successful"], cell["errors"], cell["median_ttft_ms"], flush=True)
        save(f"offload-{mode}{'-gate' if gate else ''}", report)
    return any(cell["errors"] for cell in report["cells"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("cold", "cpu", "disk"))
    parser.add_argument("--gate", action="store_true")
    args = parser.parse_args()
    raise SystemExit(int(measure(args.mode, args.gate)))
