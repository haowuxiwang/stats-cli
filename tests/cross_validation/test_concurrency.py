"""Concurrency test for stats-cli handler.

Runs 5 concurrent handler() calls via ThreadPoolExecutor.
Tests descriptive, normality, ttest commands.
Reports pass/fail, errors, and speedup ratio.
"""

import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, "D:/learn/claudecode/stats-cli")

from main import handler

# Sample data for tests
SAMPLE_VALUES = [
    23.5, 25.1, 22.8, 24.3, 26.0, 23.9, 25.5, 24.7, 23.1, 25.8,
    22.5, 24.0, 25.3, 23.7, 24.9, 26.2, 23.3, 24.6, 25.0, 23.8,
    24.1, 25.6, 23.0, 24.4, 25.2, 23.6, 24.8, 25.4, 23.2, 24.2,
]

TTEST_X = [23.5, 25.1, 22.8, 24.3, 26.0, 23.9, 25.5, 24.7, 23.1, 25.8,
           22.5, 24.0, 25.3, 23.7, 24.9, 26.2, 23.3, 24.6, 25.0, 23.8]

TTEST_Y = [21.5, 23.1, 20.8, 22.3, 24.0, 21.9, 23.5, 22.7, 21.1, 23.8,
           20.5, 22.0, 23.3, 21.7, 22.9, 24.2, 21.3, 22.6, 23.0, 21.8]

# 5 tasks: descriptive, normality, ttest, descriptive again, normality again
TASKS = [
    {"command": "descriptive", "params": {"values": SAMPLE_VALUES}},
    {"command": "normality", "params": {"values": SAMPLE_VALUES}},
    {"command": "ttest", "params": {"test_type": "two_sample", "values": TTEST_X, "values2": TTEST_Y}},
    {"command": "descriptive", "params": {"values": list(reversed(SAMPLE_VALUES))}},
    {"command": "normality", "params": {"values": [v + 0.1 for v in SAMPLE_VALUES]}},
]


def run_task(idx, task):
    """Run a single handler call, return (idx, elapsed, result_or_error, is_ok)."""
    start = time.perf_counter()
    try:
        result = handler(task)
        elapsed = time.perf_counter() - start
        is_success = result.get("status") == "success"
        return (idx, elapsed, None if is_success else result, is_success)
    except Exception as e:
        elapsed = time.perf_counter() - start
        return (idx, elapsed, f"{type(e).__name__}: {e}", False)


def run_sequential(tasks):
    """Run all tasks sequentially, return total time."""
    results = []
    start = time.perf_counter()
    for i, task in enumerate(tasks):
        r = run_task(i, task)
        results.append(r)
    total = time.perf_counter() - start
    return total, results


def run_concurrent(tasks):
    """Run all tasks concurrently via ThreadPoolExecutor, return total time."""
    results = []
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(run_task, i, task): i for i, task in enumerate(tasks)}
        for future in as_completed(futures):
            results.append(future.result())
    total = time.perf_counter() - start
    # Sort by task index for consistent reporting
    results.sort(key=lambda r: r[0])
    return total, results


def main():
    errors = []

    # --- Sequential run ---
    print("=" * 60)
    print("SEQUENTIAL RUN (5 tasks)")
    print("=" * 60)
    seq_total, seq_results = run_sequential(TASKS)
    for idx, elapsed, err, ok in seq_results:
        status = "OK" if ok else "FAIL"
        cmd = TASKS[idx]["command"]
        print(f"  Task {idx} [{cmd}]: {status} ({elapsed:.4f}s)")
        if not ok:
            errors.append(f"Sequential task {idx} ({cmd}): {err}")
    print(f"  Total sequential time: {seq_total:.4f}s\n")

    # --- Concurrent run ---
    print("=" * 60)
    print("CONCURRENT RUN (5 tasks, ThreadPoolExecutor)")
    print("=" * 60)
    conc_total, conc_results = run_concurrent(TASKS)
    for idx, elapsed, err, ok in conc_results:
        status = "OK" if ok else "FAIL"
        cmd = TASKS[idx]["command"]
        print(f"  Task {idx} [{cmd}]: {status} ({elapsed:.4f}s)")
        if not ok:
            errors.append(f"Concurrent task {idx} ({cmd}): {err}")
    print(f"  Total concurrent time: {conc_total:.4f}s\n")

    # --- Results ---
    all_ok = all(r[3] for r in seq_results) and all(r[3] for r in conc_results)
    speedup = seq_total / conc_total if conc_total > 0 else 0.0

    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    result = {
        "concurrent_pass": all_ok and len(errors) == 0,
        "errors": errors,
        "speedup_ratio": round(speedup, 2),
    }
    print(f"  concurrent_pass : {result['concurrent_pass']}")
    print(f"  errors          : {result['errors']}")
    print(f"  speedup_ratio   : {result['speedup_ratio']}")
    print(f"\n  Sequential total : {seq_total:.4f}s")
    print(f"  Concurrent total : {conc_total:.4f}s")

    return result


if __name__ == "__main__":
    main()
