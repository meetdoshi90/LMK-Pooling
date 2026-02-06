#!/usr/bin/env python3
import json
import csv
import re
from pathlib import Path
from collections import defaultdict

MLDR_LANGUAGES = ['ara', 'cmn', 'deu', 'eng', 'fra', 'hin', 'ita', 'jpn', 'kor', 'por', 'rus', 'spa', 'tha']
MIRACL_LANGUAGES = ['ara', 'ben', 'deu', 'eng', 'fas', 'fin', 'fra', 'hin', 'ind', 'jpn', 'kor', 'rus', 'spa', 'swa', 'tel', 'tha', 'yor', 'zho']
MULTIEURLEX_LANGUAGES = [
    'deu', 'eng', 'bul', 'por', 'lit', 'swe', 'est', 'spa', 'nld', 
    'pol', 'fra', 'slk', 'fin', 'mlt', 'ita', 'hun', 'lav', 'ces', 
    'ell', 'slv', 'dan', 'ron', 'hrv'
]

RESULTS_DIR = Path("results_new")
OUT_DIR = Path("results_grouped")  
DEFAULT_METRIC = "ndcg_at_10"

TASK_GROUPS = {
    "mteb_v2": [
        "ArguAna", "FiQA2018", "SCIDOCS",
        "CQADupstackUnixRetrieval",
        "CQADupstackGamingRetrieval",
        "ClimateFEVERHardNegatives",
        "FEVERHardNegatives",
        "HotpotQAHardNegatives",
        "TRECCOVID",
        "Touche2020Retrieval.v3",
    ],
    "beir-15": [
        "NFCorpus", "NQ", "HotpotQA", "Touche2020",
        "CQADupstackRetrieval", "QuoraRetrieval",
        "DBPedia", "FEVER", "ClimateFEVER", "SciFact",
        "TRECCOVID", "FiQA2018", "ArguAna", "SCIDOCS", "MSMARCO",
    ],
    "mldr": ["MultiLongDocRetrieval"],
    "miracl_hn": ["MIRACLRetrievalHardNegatives"],
    "multieurlex": ["MultiEURLEXMultilabelClassification"],
    "msmarco": ["MSMARCO"],
    "long_embed": [
        "LEMBNeedleRetrieval",
        "LEMBPasskeyRetrieval",
        "LEMBQMSumRetrieval",
        "LEMBSummScreenFDRetrieval",
        "LEMBWikimQARetrieval",
        "LEMBNarrativeQARetrieval",
    ],
    "coir": [
        "AppsRetrieval",
        "CodeFeedbackMT",
        "CodeFeedbackST",
        "CodeTransOceanContest",
        "CodeTransOceanDL",
        "CosQA",
        "SyntheticText2SQL",
        "StackOverflowQA",
        "COIRCodeSearchNetRetrieval",
        "CodeSearchNetCCRetrieval",
    ],
}

TASK_METRICS = {
    "ArguAna": "ndcg_at_10",
    "FiQA2018": "ndcg_at_10",
    "SCIDOCS": "ndcg_at_10",
    "CQADupstackUnixRetrieval": "ndcg_at_10",
    "CQADupstackGamingRetrieval": "ndcg_at_10",
    "ClimateFEVERHardNegatives": "ndcg_at_10",
    "FEVERHardNegatives": "ndcg_at_10",
    "HotpotQAHardNegatives": "ndcg_at_10",
    "TRECCOVID": "ndcg_at_10",
    "Touche2020Retrieval.v3": "ndcg_at_10",
    "NFCorpus": "ndcg_at_10",
    "NQ": "ndcg_at_10",
    "HotpotQA": "ndcg_at_10",
    "Touche2020": "ndcg_at_10",
    "CQADupstackRetrieval": "ndcg_at_10",
    "QuoraRetrieval": "ndcg_at_10",
    "DBPedia": "ndcg_at_10",
    "FEVER": "ndcg_at_10",
    "ClimateFEVER": "ndcg_at_10",
    "SciFact": "ndcg_at_10",
    "MIRACLRetrievalHardNegatives": "ndcg_at_10",
    "MultiEURLEXMultilabelClassification": "f1",
    "MSMARCO": "ndcg_at_10",
    "MultiLongDocRetrieval": "ndcg_at_10",
    "LEMBNeedleRetrieval": "precision_at_1",
    "LEMBPasskeyRetrieval": "precision_at_1",
    "LEMBQMSumRetrieval": "ndcg_at_10",
    "LEMBSummScreenFDRetrieval": "ndcg_at_10",
    "LEMBWikimQARetrieval": "ndcg_at_10",
    "LEMBNarrativeQARetrieval": "ndcg_at_10",
    "AppsRetrieval": "ndcg_at_10",
    "CodeFeedbackMT": "ndcg_at_10",
    "CodeFeedbackST": "ndcg_at_10",
    "CodeTransOceanContest": "ndcg_at_10",
    "CodeTransOceanDL": "ndcg_at_10",
    "CosQA": "ndcg_at_10",
    "SyntheticText2SQL": "ndcg_at_10",
    "StackOverflowQA": "ndcg_at_10",
    "COIRCodeSearchNetRetrieval": "ndcg_at_10",
    "CodeSearchNetCCRetrieval": "ndcg_at_10",
}
def load_and_average_metric(json_path: Path, metric: str):
    with json_path.open() as f:
        data = json.load(f)
    scores = []
    keys = list(data.keys())
    # if len(keys)!=1:
    #     raise KeyError(keys)
    # if len(data.get(keys[0], []))>1:
    #     print(json_path, len(data.get(keys[0], [])))
    for key in keys:
        for entry in data.get(key, []):
            if metric in entry and entry[metric] is not None:
                scores.append(entry[metric])
    if not scores:
        raise ValueError(f"{metric} not found in {json_path}")
    return sum(scores) / len(scores)

task_to_group = {}
for g, tasks in TASK_GROUPS.items():
    for t in tasks:
        if t not in task_to_group:
            task_to_group[t] = []
        task_to_group[t].append(g)

group_rows = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

# regex to locate: _<True|False>_<split_size_digits>_<max_seq_len_digits>_
pattern = re.compile(r'_(True|False)_(\d+)_(\d+)_')

for d in sorted(RESULTS_DIR.iterdir()):
    if not d.is_dir():
        print(f"[WARN] skipping (is not a dir): {folder_name}")
        continue
    folder_name = d.name

    m = pattern.search(folder_name)
    if not m:
        print(f"[WARN] skipping folder (pattern not found): {folder_name}")
        continue

    fixed_str = m.group(1)
    split_size = m.group(2)
    max_seq_len = m.group(3)

    prefix = folder_name[: m.start()]   
    model_part = folder_name[m.end():]  

    if "_" not in prefix:
        print(f"[WARN] can't split task/pool for folder: {folder_name} (no '_' before pool)")
        continue
    
    if any([prefix.startswith(x) for x in MLDR_LANGUAGES + MIRACL_LANGUAGES + MULTIEURLEX_LANGUAGES]):
        lang, task_name, pool = prefix.split("_", 2)
    else:
        lang = 'eng' 
        task_name, pool = prefix.split("_", 1) 

    if task_name not in task_to_group:
        print(f"[INFO] skipping unknown task folder: {folder_name} (task '{task_name}' not in known TASK_GROUPS)")
        continue

    groups = task_to_group[task_name]
    metric = TASK_METRICS.get(task_name, DEFAULT_METRIC)

    json_path = d / f"{task_name}.json"
    if not json_path.exists():
        print(f"[WARN] missing json for {folder_name}: {json_path}")
        continue

    try:
        avg_score = load_and_average_metric(json_path, metric)
    except Exception as e:
        print(f"[WARN] failed to read {metric} from {json_path}: {e}")
        continue

    config_key = (model_part, pool, fixed_str, int(split_size), int(max_seq_len))
    for group in groups:
        if group in ['mldr','miracl_hn','multieurlex']:
            task_name = lang
        group_rows[group][config_key][task_name].append(avg_score)

for group, configs in group_rows.items():
    tasks_in_group = sorted(TASK_GROUPS[group])
    if group == 'mldr':
        tasks_in_group = MLDR_LANGUAGES
    elif group == 'miracl_hn':
        tasks_in_group = MIRACL_LANGUAGES
    elif group == 'multieurlex':
        tasks_in_group = MULTIEURLEX_LANGUAGES
    
    headers = (
        ["model_name", "pool", "fixed_splitter", "split_size", "max_seq_len"]
        + tasks_in_group
        + ["avg_tasks"]
    )

    out_csv = OUT_DIR / f"{group}_summary.csv"
    with out_csv.open("w", newline="") as csvf:
        writer = csv.writer(csvf)
        writer.writerow(headers)

        for config_key, task_scores_dict in sorted(configs.items()):
            model_name, pool, fixed_str, split_size, max_seq_len = config_key
            row = [model_name, pool, fixed_str, split_size, max_seq_len]

            group_vals = []

            for task in tasks_in_group:
                vals = task_scores_dict.get(task, [])
                if not vals:
                    row.append("")
                else:
                    mean_val = sum(vals) / len(vals)
                    row.append(f"{mean_val:.6f}")
                    group_vals.append(mean_val)

            if len(group_vals) == len(tasks_in_group):
                row.append(f"{(sum(group_vals) / len(group_vals)):.6f}")
            else:
                row.append("")

            writer.writerow(row)

    print(f"[OK] wrote {out_csv} ({len(configs)} rows, {len(tasks_in_group)} task cols)")

print("Done.")
