"""Compare the baseline Python report with the Go client on one cache-off server."""

import argparse
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

from bench_phase2 import LEVELS, RESULTS, ROOT, URL, save, server, utc_now

REFERENCE = ROOT / "results" / "vllm.json"


def compare(reports):
    baseline = json.loads(REFERENCE.read_text(encoding="utf-8"))
    rows = []
    for index, concurrency in enumerate(LEVELS):
        reference = baseline["runs"][index]
        order = [(item["id"], item["expected"]) for item in reference["outcomes"]]
        blocks = {client: [report["runs"][index] for report in passes]
                  for client, passes in reports.items()}
        all_blocks = blocks["python"] + blocks["go"]
        checks = {
            "matching_top_level_schema": all(set(report) == set(baseline)
                                             for passes in reports.values() for report in passes),
            "matching_inputs": all(report[key] == baseline[key]
                                   for passes in reports.values() for report in passes
                                   for key in ("dataset_sha256", "shuffle_seed", "rounds_per_level",
                                               "model_alias", "max_output_tokens")),
            "matching_outcome_schema": all(set(block) == set(reference)
                                           and all(set(item) == set(reference["outcomes"][0])
                                                   for item in block["outcomes"]) for block in all_blocks),
            "matching_workload_order": all(block["concurrency"] == concurrency
                                           and [(item["id"], item["expected"]) for item in block["outcomes"]] == order
                                           for block in all_blocks),
            "matching_routing_counts": len({block["correct"] for block in all_blocks}) == 1,
            "zero_errors": all(block["errors"] == 0 for block in all_blocks),
        }
        medians = {}
        for metric in ("ttft_ms", "latency_ms"):
            values = {client: [item[metric] for block in passes for item in block["outcomes"]
                               if "error" not in item] for client, passes in blocks.items()}
            if not all(values.values()):
                medians[metric] = {"passes_tolerance": False}
                continue
            py, go = (statistics.median(values[client]) for client in ("python", "go"))
            delta, tolerance = go - py, max(py * 0.10, 10)
            medians[metric] = {"python": round(py, 3), "go": round(go, 3),
                               "go_minus_python": round(delta, 3),
                               "tolerance": round(tolerance, 3), "passes_tolerance": abs(delta) <= tolerance}
        rows.append({"concurrency": concurrency, "checks": checks, "medians_ms": medians,
                     "route_counts": {client: [block["correct"] for block in passes]
                                      for client, passes in blocks.items()}})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("go_binary", type=Path)
    args = parser.parse_args()
    go_binary = args.go_binary.resolve(strict=True)
    order = ("python", "go", "go", "python")
    reports = {"go": [], "python": []}
    go_version = subprocess.check_output(
        ["go", "version", "-m", str(go_binary)], text=True
    ).splitlines()[0].split()[-1]
    commands = {"python": (sys.executable, "-B", "benchmark.py"),
                "go": (str(go_binary),)}
    with server("vllm", "off") as (metadata, _):
        runs = []
        for client in order:
            name = f"go-parity-{client}-{1 + len(reports[client])}.json"
            path = RESULTS / name
            if path.exists():
                raise FileExistsError(path)
            completed = subprocess.run(
                [*commands[client], "vllm", "--url", URL, "--output", str(path)], cwd=ROOT,
            )
            if not path.exists():
                raise RuntimeError(f"{client} failed before writing the benchmark report (exit={completed.returncode})")
            runs.append({"client": client, "file": name, "exit_code": completed.returncode})
            reports[client].append(json.loads(path.read_text(encoding="utf-8")))
        rows = compare(reports)
        save("go-parity", {"at": utc_now(), "server": metadata, "go": go_version,
                           "python": sys.version.split()[0], "runs": runs, "comparison": rows})
    print(json.dumps(rows, indent=2))
    return int(any(not all(row["checks"].values())
                   or not all(metric["passes_tolerance"] for metric in row["medians_ms"].values())
                   for row in rows))


if __name__ == "__main__":
    raise SystemExit(main())
