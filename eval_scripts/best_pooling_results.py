#!/usr/bin/env python3
"""
best_pooling_results.py

Reads decoder_results_grouped/*_summary.csv and reports best performance
per pooling type per checkpoint.

Rules:
  mean / last : use avg_tasks directly (no variable hyperparameter)
  lmk         : for each task column take the max across all granularities,
                then average those per-task bests → "oracle lmk"
                (different tasks can benefit from different granularities)

Outputs one CSV per group + a console comparison table.
"""

import csv
import argparse
from pathlib import Path
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("--results_dir", default="decoder_results_grouped")
ap.add_argument("--out_dir",     default="decoder_best_results")
args = ap.parse_args()

RESULTS_DIR = Path(args.results_dir)
OUT_DIR     = Path(args.out_dir)
OUT_DIR.mkdir(parents=True, exist_ok=True)


def to_float(s):
    try:    return float(s)
    except: return None


META_COLS = {"model_name", "pool", "lmk_granularity", "prompt", "max_seq_len", "avg_tasks"}

for csv_path in sorted(RESULTS_DIR.glob("*_summary.csv")):
    group = csv_path.stem.replace("_summary", "")

    with csv_path.open() as f:
        reader    = csv.DictReader(f)
        headers   = reader.fieldnames or []
        all_rows  = list(reader)

    if not all_rows:
        continue

    task_cols = [h for h in headers if h not in META_COLS]

    # ── Group rows by (model_name, prompt, max_seq_len, pool) ─────────────────
    grouped = defaultdict(list)
    for row in all_rows:
        key = (row["model_name"], row["prompt"], row["max_seq_len"], row["pool"])
        grouped[key].append(row)

    # ── Compute best score per (model, prompt, msl, pool) ─────────────────────
    results = []

    for (model_name, prompt, max_seq_len, pool), rows in grouped.items():

        if pool in ("mean", "last"):
            # No variable factor: just pick the row with the highest avg_tasks
            valid = [(to_float(r["avg_tasks"]), r) for r in rows
                     if to_float(r.get("avg_tasks")) is not None]
            if not valid:
                continue
            best_avg, best_row = max(valid, key=lambda x: x[0])
            results.append({
                "model_name":     model_name,
                "pool":           pool,
                "lmk_granularity": "",
                "prompt":         prompt,
                "max_seq_len":    max_seq_len,
                "task_scores":    {t: to_float(best_row.get(t)) for t in task_cols},
                "avg_tasks":      best_avg,
            })

        elif pool == "lmk":
            # For every task independently take the best granularity's score
            per_task_best = {}
            for task in task_cols:
                best_val = max(
                    (to_float(r.get(task)) for r in rows),
                    default=None,
                    key=lambda v: v if v is not None else -1,
                )
                if best_val is not None:
                    per_task_best[task] = best_val

            if not per_task_best:
                continue

            complete = len(per_task_best) == len(task_cols)
            avg      = (sum(per_task_best.values()) / len(task_cols)) if complete else None

            results.append({
                "model_name":     model_name,
                "pool":           "lmk",
                "lmk_granularity": "best_per_task",
                "prompt":         prompt,
                "max_seq_len":    max_seq_len,
                "task_scores":    {t: per_task_best.get(t) for t in task_cols},
                "avg_tasks":      avg,
                "n_tasks_found":  len(per_task_best),
                "n_tasks_total":  len(task_cols),
            })

    if not results:
        print(f"[SKIP] {group}: no parseable results")
        continue

    # ── Write output CSV ───────────────────────────────────────────────────────
    out_path    = OUT_DIR / f"{group}_best.csv"
    out_headers = (["model_name","pool","lmk_granularity","prompt","max_seq_len"]
                   + task_cols + ["avg_tasks"])

    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(out_headers)
        for r in sorted(results, key=lambda x: (x["pool"], x["model_name"])):
            row = [r["model_name"], r["pool"], r["lmk_granularity"],
                   r["prompt"], r["max_seq_len"]]
            for t in task_cols:
                v = r["task_scores"].get(t)
                row.append(f"{v:.6f}" if v is not None else "")
            if r["avg_tasks"] is not None:
                row.append(f"{r['avg_tasks']:.6f}")
            else:
                row.append(f"partial_{r.get('n_tasks_found',0)}/{r.get('n_tasks_total', len(task_cols))}")
            writer.writerow(row)

    # ── Console comparison table ───────────────────────────────────────────────
    print(f"\n{'='*90}")
    print(f"  {group.upper()}  |  {len(task_cols)} tasks  |  {out_path}")
    print(f"{'='*90}")

    # Per-pool ranking
    for pool_type in ("mean", "last", "lmk"):
        pool_res = [r for r in results
                    if r["pool"] == pool_type and r["avg_tasks"] is not None]
        if not pool_res:
            continue
        pool_res.sort(key=lambda x: x["avg_tasks"], reverse=True)

        gran_note = "  (best granularity selected per task)" if pool_type == "lmk" else ""
        print(f"\n  ── {pool_type.upper()}{gran_note}")
        print(f"  {'avg':>7}  {'prompt':<10} {'msl':<7}  model")
        print(f"  {'-'*80}")
        for r in pool_res:
            print(f"  {r['avg_tasks']:>7.4f}  {r['prompt']:<10} {r['max_seq_len']:<7}  {r['model_name']}")

    # Side-by-side best-per-pool comparison (same model_name only)
    # Group by (model_name, prompt, max_seq_len) and compare pools
    by_model: dict = defaultdict(dict)
    for r in results:
        if r["avg_tasks"] is None:
            continue
        mk = (r["model_name"], r["prompt"], r["max_seq_len"])
        p  = r["pool"]
        # Keep best avg per pool for this model config
        if p not in by_model[mk] or r["avg_tasks"] > by_model[mk][p]:
            by_model[mk][p] = r["avg_tasks"]

    if by_model:
        print(f"\n  ── Side-by-side (same checkpoint)")
        print(f"  {'mean':>7}  {'last':>7}  {'lmk (oracle)':>12}  {'prompt':<10} {'msl':<7}  model")
        print(f"  {'-'*90}")
        for (model_name, prompt, msl), pool_avgs in sorted(by_model.items()):
            mean_s = f"{pool_avgs['mean']:>7.4f}" if 'mean' in pool_avgs else f"{'--':>7}"
            last_s = f"{pool_avgs['last']:>7.4f}" if 'last' in pool_avgs else f"{'--':>7}"
            lmk_s  = f"{pool_avgs['lmk']:>12.4f}" if 'lmk'  in pool_avgs else f"{'--':>12}"
            print(f"  {mean_s}  {last_s}  {lmk_s}  {prompt:<10} {msl:<7}  {model_name}")

print("\nDone.")