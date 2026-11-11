"""Run the same labeled support-ticket workload against either engine."""

import argparse
import hashlib
import json
import math
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from triage import DEFAULT_URL, MAX_TOKENS, MODEL, QUEUES, TriageError, triage

TICKETS = Path(__file__).with_name("tickets.jsonl")
SEED = 5090
WARMUP_TICKET = "The April invoice lists our subscription twice. Please correct it."


def p95(values):
    ordered = sorted(values)
    return round(ordered[math.ceil(0.95 * len(ordered)) - 1], 2) if ordered else None


def measure(tickets, concurrency, url):
    def send(ticket):
        outcome = {"id": ticket["id"], "expected": ticket["queue"]}
        try:
            outcome.update(asdict(triage(ticket["text"], url)))
        except TriageError as error:
            outcome["error"] = str(error)
        return outcome

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        started_at = time.perf_counter()
        outcomes = list(pool.map(send, tickets))
        elapsed = time.perf_counter() - started_at
    successful = [outcome for outcome in outcomes if "error" not in outcome]
    correct = sum(outcome["queue"] == outcome["expected"] for outcome in successful)
    return {
        "concurrency": concurrency,
        "requests": len(outcomes),
        "successful": len(successful),
        "errors": len(outcomes) - len(successful),
        "correct": correct,
        "accuracy_pct": round(100 * correct / len(outcomes), 2),
        "wall_seconds": round(elapsed, 3),
        "successful_requests_per_second": round(len(successful) / elapsed, 2),
        "output_tokens_per_second": round(
            sum(outcome["completion_tokens"] for outcome in successful) / elapsed, 2
        ),
        "p95_ttft_ms": p95(outcome["ttft_ms"] for outcome in successful),
        "p95_latency_ms": p95(outcome["latency_ms"] for outcome in successful),
        "outcomes": outcomes,
    }


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("Must be a positive integer")
    return number


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("engine", choices=("vllm", "sglang"), help="Label for the active server")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--concurrency", nargs="+", type=positive_int, default=[1, 4, 8])
    parser.add_argument("--rounds", type=positive_int, default=2)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    dataset = TICKETS.read_bytes()
    tickets = [json.loads(line) for line in dataset.splitlines()]
    ids = {ticket["id"] for ticket in tickets}
    if not tickets or len(ids) != len(tickets) or any(
        ticket["queue"] not in QUEUES or not ticket["text"].strip() for ticket in tickets
    ):
        parser.error("tickets.jsonl must have unique IDs, nonempty text, and known queue labels")

    rng = random.Random(SEED)
    workload = []
    for _ in range(args.rounds):
        shuffled = tickets.copy()
        rng.shuffle(shuffled)
        workload.extend(shuffled)

    runs = []
    started_at = datetime.now(timezone.utc).isoformat()
    for concurrency in args.concurrency:
        try:
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                list(pool.map(lambda _: triage(WARMUP_TICKET, args.url), range(concurrency)))
        except TriageError as error:
            print(f"Warmup failed: {error}", file=sys.stderr)
            return 1
        run = measure(workload, concurrency, args.url)
        runs.append(run)
        print(json.dumps({key: value for key, value in run.items() if key != "outcomes"}), flush=True)

    report = {
        "engine": args.engine,
        "url": args.url,
        "model_alias": MODEL,
        "started_at_utc": started_at,
        "dataset_sha256": hashlib.sha256(dataset).hexdigest(),
        "shuffle_seed": SEED,
        "rounds_per_level": args.rounds,
        "max_output_tokens": MAX_TOKENS,
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return int(any(run["errors"] for run in runs))


if __name__ == "__main__":
    sys.exit(main())
