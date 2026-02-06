import os
import json
import csv
from collections import defaultdict


ROOT_DIR = "mteb_results"
TASK_CSV = "results_grouped/mteb_task_scores.csv"
MERGED_CSV = "results_grouped/mteb_merged_scores.csv"

TASK_GROUPS = {
    "Classification": [
        "AmazonCounterfactualClassification",
        "Banking77Classification",
        "ImdbClassification",
        "MTOPDomainClassification",
        "MassiveIntentClassification",
        "MassiveScenarioClassification",
        "ToxicConversationsClassification",
        "TweetSentimentExtractionClassification",
    ],
    "Clustering": [
        "ArXivHierarchicalClusteringP2P",
        "ArXivHierarchicalClusteringS2S",
        "BiorxivClusteringP2P.v2",
        "MedrxivClusteringP2P.v2",
        "MedrxivClusteringS2S.v2",
        "StackExchangeClustering.v2",
        "StackExchangeClusteringP2P.v2",
        "TwentyNewsgroupsClustering.v2",
    ],
    "PairClassification": [
        "SprintDuplicateQuestions",
        "TwitterSemEval2015",
        "TwitterURLCorpus",
    ],
    "Reranking": [
        "AskUbuntuDupQuestions",
        "MindSmallReranking",
    ],
    "Retrieval": [
        "ArguAna",
        "ClimateFEVERHardNegatives",
        "CQADupstackGamingRetrieval",
        "CQADupstackUnixRetrieval",
        "FEVERHardNegatives",
        "FiQA2018",
        "HotpotQAHardNegatives",
        "SCIDOCS",
        "Touche2020Retrieval.v3",
        "TRECCOVID",
    ],
    "STS": [
        "BIOSSES",
        "SICK-R",
        "STS12",
        "STS13",
        "STS14",
        "STS15",
        "STS17",
        "STS22.v2",
        "STSBenchmark",
    ],
    "Summarization": [
        "SummEvalSummarization.v2",
    ],
}

ALL_TASKS = sorted({t for v in TASK_GROUPS.values() for t in v})

def load_main_score(path):
    with open(path, "r") as f:
        data = json.load(f)

    entries = data.get("test", [])
    scores = [e["main_score"] for e in entries if "main_score" in e]

    if not scores:
        return None

    if len(scores) > 1:
        print(f"[INFO] Multiple scores found in {path}, averaging.")

    return sum(scores) / len(scores) * 100.0


task_scores = defaultdict(dict)
merged_scores = {}

for model in sorted(os.listdir(ROOT_DIR)):
    model_path = os.path.join(ROOT_DIR, model)
    if not os.path.isdir(model_path):
        continue

    merged_scores[model] = {}
    all_vals = []

    for task in ALL_TASKS:
        json_path = os.path.join(model_path, f"{task}.json")

        if not os.path.exists(json_path):
            raise Exception(f'{task} file not found')
            task_scores[model][task] = ""
            continue

        val = load_main_score(json_path)
        task_scores[model][task] = round(val, 2)
        all_vals.append(val)


    for group, tasks in TASK_GROUPS.items():
        vals = [
            task_scores[model][t]
            for t in tasks
            if task_scores[model][t] != ""
        ]
        merged_scores[model][f"Avg_{group}"] = (
            round(sum(vals) / len(vals), 2) if vals else ""
        )

    merged_scores[model]["Avg_All_MTEB"] = (
        round(sum(all_vals) / len(all_vals), 2) if all_vals else ""
    )


with open(TASK_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["model"] + ALL_TASKS)

    for model in sorted(task_scores):
        writer.writerow(
            [model] + [task_scores[model].get(t, "") for t in ALL_TASKS]
        )


merged_columns = (
    [f"Avg_{k}" for k in TASK_GROUPS.keys()] + ["Avg_All_MTEB"]
)

with open(MERGED_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["model"] + merged_columns)

    for model in sorted(merged_scores):
        writer.writerow(
            [model] + [merged_scores[model].get(c, "") for c in merged_columns]
        )

print(f"\nSaved:\n  {TASK_CSV}\n  {MERGED_CSV}")
