#!/usr/bin/env python3
"""
aggregate_decoder_results.py

Parses decoder_eval_results/ folders produced by eval_decoder.py
and writes per-task-group CSV summaries to decoder_results_grouped/.

Folder name format (from build_run_tag() in eval_decoder.py):
    {task_name}_{lang}_{pool_tag}_{prompt_tag}_{max_seq_len}_{model_tag}

    pool_tag   : lmk_gran{N}  |  mean  |  last
    prompt_tag : prompt  |  noprompt
    model_tag  : last 3 path components of checkpoint joined by _

Run:
    python aggregate_decoder_results.py
    python aggregate_decoder_results.py --results_dir /path/to/decoder_eval_results
"""

import json
import csv
import re
import argparse
from pathlib import Path
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("--results_dir", default="decoder_eval_results")
ap.add_argument("--out_dir",     default="decoder_results_grouped")
script_args = ap.parse_args()

RESULTS_DIR    = Path(script_args.results_dir)
OUT_DIR        = Path(script_args.out_dir)
DEFAULT_METRIC = "ndcg_at_10"

# ── Task groups ────────────────────────────────────────────────────────────────
TASK_GROUPS = {
    "mteb_v2": [
        "ArguAna", "FiQA2018", "SCIDOCS",
        "CQADupstackUnixRetrieval", "CQADupstackGamingRetrieval",
        "ClimateFEVERHardNegatives", "FEVERHardNegatives",
        "HotpotQAHardNegatives", "TRECCOVID", "Touche2020Retrieval.v3",
    ],
    "beir-15": [
        "NFCorpus", "NQ", "HotpotQA", "Touche2020",
        "CQADupstackRetrieval", "QuoraRetrieval",
        "DBPedia", "FEVER", "ClimateFEVER", "SciFact",
        "TRECCOVID", "FiQA2018", "ArguAna", "SCIDOCS", "MSMARCO",
    ],
    "msmarco":    ["MSMARCO"],
    "mldr":       ["MultiLongDocRetrieval"],
    "miracl_hn":  ["MIRACLRetrievalHardNegatives"],
    "long_embed": [
        "LEMBNeedleRetrieval", "LEMBPasskeyRetrieval",
        "LEMBQMSumRetrieval", "LEMBSummScreenFDRetrieval",
        "LEMBWikimQARetrieval", "LEMBNarrativeQARetrieval",
    ],
    "coir": [
        "AppsRetrieval", "CodeFeedbackMT", "CodeFeedbackST",
        "CodeTransOceanContest", "CodeTransOceanDL", "CosQA",
        "SyntheticText2SQL", "StackOverflowQA",
        "COIRCodeSearchNetRetrieval", "CodeSearchNetCCRetrieval",
    ],
}

# Per-task metrics (LEMB needle/passkey use precision@1, everything else ndcg@10)
TASK_METRICS = {
    "LEMBNeedleRetrieval":   "precision_at_1",
    "LEMBPasskeyRetrieval":  "precision_at_1",
}   # all unlisted tasks → DEFAULT_METRIC

MLDR_LANGUAGES   = ['ara','cmn','deu','eng','fra','hin','ita','jpn','kor','por','rus','spa','tha']
MIRACL_LANGUAGES = ['ara','ben','deu','eng','fas','fin','fra','hin','ind','jpn','kor','rus','spa',
                    'swa','tel','tha','yor','zho']

# Multilingual groups: column = language code, not task name
MULTILINGUAL_GROUPS = {'mldr': MLDR_LANGUAGES, 'miracl_hn': MIRACL_LANGUAGES}


# ── Score loading ──────────────────────────────────────────────────────────────

def load_score(json_path: Path, metric: str) -> float:
    """
    Load a metric from an MTEB result JSON.
    Averages across all hf_subsets and splits found (handles CQADupstack etc.).
    """
    with json_path.open() as f:
        data = json.load(f)
    scores = []
    # data is either {split: [entry, ...]} or [[{split: [...]}], ...]
    if isinstance(data, list):
        # list of subset dicts
        for subset in data:
            for split_entries in subset.values():
                for entry in split_entries:
                    if metric in entry and entry[metric] is not None:
                        scores.append(entry[metric])
    else:
        for split_entries in data.values():
            for entry in split_entries:
                if metric in entry and entry[metric] is not None:
                    scores.append(entry[metric])
    if not scores:
        raise ValueError(f"metric '{metric}' not found in {json_path}")
    return sum(scores) / len(scores)


# ── Folder name parser ─────────────────────────────────────────────────────────
# Sort all known task names longest-first so longer names (e.g. CQADupstackUnixRetrieval)
# are checked before shorter prefixes (e.g. CQADupstackRetrieval).
_ALL_TASKS = sorted(
    {t for tasks in TASK_GROUPS.values() for t in tasks},
    key=len, reverse=True,
)

_FOLDER_RE = re.compile(
    r'^([a-z]+)'               # lang (e.g. eng, ara)
    r'_(lmk_gran\d+|mean|last)'  # pool_tag
    r'_(prompt|noprompt)'      # prompt_tag
    r'_(\d+)'                  # max_seq_len
    r'_(.+)$'                  # model_tag (everything else)
)

def parse_folder(folder_name: str):
    """
    Returns (task_name, lang, pool, lmk_gran, prompt_tag, max_seq_len, model_tag)
    or None if the folder name doesn't match.
    """
    for task in _ALL_TASKS:
        if not folder_name.startswith(task + "_"):
            continue
        rest = folder_name[len(task) + 1:]
        m = _FOLDER_RE.match(rest)
        if not m:
            continue
        lang, pool_tag, prompt_tag, msl_str, model_tag = m.groups()
        gran_m   = re.match(r'lmk_gran(\d+)', pool_tag)
        pool     = 'lmk' if gran_m else pool_tag
        lmk_gran = int(gran_m.group(1)) if gran_m else None
        return task, lang, pool, lmk_gran, prompt_tag, int(msl_str), model_tag
    return None


# ── Aggregation ────────────────────────────────────────────────────────────────

task_to_groups = defaultdict(list)
for g, tasks in TASK_GROUPS.items():
    for t in tasks:
        if g not in task_to_groups[t]:
            task_to_groups[t].append(g)

# group_rows[group][config_key][col_key] = [score, ...]
group_rows: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

OUT_DIR.mkdir(parents=True, exist_ok=True)

for d in sorted(RESULTS_DIR.iterdir()):
    if not d.is_dir():
        continue
    parsed = parse_folder(d.name)
    if parsed is None:
        print(f"[WARN] skipping (parse failed): {d.name}")
        continue

    task_name, lang, pool, lmk_gran, prompt_tag, max_seq_len, model_tag = parsed

    groups = task_to_groups.get(task_name)
    if not groups:
        print(f"[INFO] skipping unknown task '{task_name}': {d.name}")
        continue

    metric    = TASK_METRICS.get(task_name, DEFAULT_METRIC)
    json_path = d / f"{task_name}.json"
    if not json_path.exists():
        print(f"[WARN] missing json: {json_path}")
        continue

    try:
        score = load_score(json_path, metric)
    except Exception as e:
        print(f"[WARN] {json_path}: {e}")
        continue

    # config_key uniquely identifies a model + pooling + prompt + seq-len combination
    config_key = (model_tag, pool, lmk_gran, prompt_tag, max_seq_len)

    for group in groups:
        # For multilingual groups the column is the language code; otherwise the task name
        col_key = lang if group in MULTILINGUAL_GROUPS else task_name
        group_rows[group][config_key][col_key].append(score)
        print(f"[READ] {group:12s} | {task_name:35s} | {pool:8s} gran={str(lmk_gran):4s} "
              f"| {prompt_tag:8s} | msl={max_seq_len} | {model_tag[-40:]} → {score:.4f}")


# ── Write CSVs ────────────────────────────────────────────────────────────────

for group, configs in sorted(group_rows.items()):
    task_cols = MULTILINGUAL_GROUPS.get(group, sorted(TASK_GROUPS[group]))

    headers = (
        ["model_name", "pool", "lmk_granularity", "prompt", "max_seq_len"]
        + task_cols
        + ["avg_tasks"]
    )

    out_csv = OUT_DIR / f"{group}_summary.csv"
    with out_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)

        for config_key in sorted(configs.keys()):
            model_tag, pool, lmk_gran, prompt_tag, max_seq_len = config_key
            task_scores = configs[config_key]

            row = [
                model_tag,
                pool,
                lmk_gran if lmk_gran is not None else "",
                prompt_tag,
                max_seq_len,
            ]
            valid_scores = []
            for col in task_cols:
                vals = task_scores.get(col, [])
                if vals:
                    v = sum(vals) / len(vals)
                    row.append(f"{v:.6f}")
                    valid_scores.append(v)
                else:
                    row.append("")

            # avg_tasks only when all columns are present
            if len(valid_scores) == len(task_cols):
                row.append(f"{sum(valid_scores)/len(valid_scores):.6f}")
            else:
                row.append(f"partial_{len(valid_scores)}/{len(task_cols)}")

            writer.writerow(row)

    print(f"[OK] {out_csv}  ({len(configs)} configs × {len(task_cols)} tasks)")

print("Done.")