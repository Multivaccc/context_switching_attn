import os
import json
import csv
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def save_results(records: list, json_path: str, csv_path: str):
    os.makedirs(os.path.dirname(json_path), exist_ok=True)

    with open(json_path, "w") as f:
        json.dump(records, f, indent=2)

    if records:
        keys = list(records[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for rec in records:
                writer.writerow(rec)

def plot_degradation(records: list, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    data = defaultdict(list)

    for r in records:
        data[(r["phase"], r["task"])].append(r)

    for (phase, task), recs in data.items():
        recs.sort(key=lambda x: x["history_length"])
        xs = [r["history_length"] for r in recs]
        ys = [r.get("accuracy", r.get("rouge1", 0)) for r in recs]
        plt.figure()
        plt.plot(xs, ys, label=f"{phase}")
        plt.xlabel("History length")
        plt.ylabel("Metric")
        plt.title(f"{task}")
        plt.legend()
        plt.savefig(f"{out_dir}/{task}_{phase}.png")
        plt.close()

def plot_tau_matrix(records: list, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    tasks = sorted({r["src_task"] for r in records if "src_task" in r})
    n = len(tasks)
    mat = np.zeros((n, n))

    for r in records:
        if r.get("phase") == "base" and "src_task" in r:
            i = tasks.index(r["src_task"])
            j = tasks.index(r["tgt_task"])
            mat[i, j] = r["tau"]

    plt.figure(figsize=(6, 6))
    plt.imshow(mat, cmap="viridis")
    plt.xticks(range(n), tasks, rotation=90)
    plt.yticks(range(n), tasks)
    plt.colorbar(label="tau")
    plt.title("Tau matrix")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/tau_matrix.png")
    plt.close()