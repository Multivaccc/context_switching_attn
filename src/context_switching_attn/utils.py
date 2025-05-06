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
    if not records:
        return
    # union of keys in order seen
    all_keys = []
    for rec in records:
        for k in rec:
            if k not in all_keys:
                all_keys.append(k)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        for rec in records:
            writer.writerow(rec)

def plot_base_degradation(base_records: list, out_dir: str, model_name: str = None):
    os.makedirs(out_dir, exist_ok=True)
    # group by task
    by_task = defaultdict(list)
    for r in base_records:
        if r.get("phase") == "base" and "task" in r:
            by_task[r["task"]].append(r)

    for task, recs in by_task.items():
        # sort by history_length
        recs = sorted(recs, key=lambda x: x["history_length"])
        # pick metric in order of preference
        for m in ("accuracy","exact_match","rouge1","tau"):
            if any(m in r for r in recs):
                metric = m
                break
        else:
            continue
        plt.figure()
        xs = [r["history_length"] for r in recs]
        ys = [r[metric] for r in recs]
        plt.plot(xs, ys, marker="o", label=metric)
        # shade if se present
        se_key = metric + "_se"
        if all(se_key in r for r in recs):
            ses = [r[se_key] for r in recs]
            lower = [y - s for y, s in zip(ys, ses)]
            upper = [y + s for y, s in zip(ys, ses)]
            plt.fill_between(xs, lower, upper, alpha=0.2)
        plt.xlabel("History length")
        plt.ylabel(metric)
        title = f"{task} — {metric} (base)"
        if model_name:
            title = f"{model_name}: " + title
        plt.title(title)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{task}_base_degradation.png"))
        plt.close()

def plot_tau_matrix(records: list, out_dir: str, model_name: str = None):
    os.makedirs(out_dir, exist_ok=True)
    tasks = sorted({r["src_task"] for r in records if r.get("phase")=="base" and "src_task" in r})
    n = len(tasks)
    mat = np.zeros((n, n))
    for r in records:
        if r.get("phase")=="base" and "src_task" in r:
            i = tasks.index(r["src_task"])
            j = tasks.index(r["tgt_task"])
            mat[i, j] = r["tau"]
    plt.figure(figsize=(6,6))
    plt.imshow(mat, cmap="viridis", aspect="equal")
    plt.xticks(range(n), tasks, rotation=90)
    plt.yticks(range(n), tasks)
    plt.colorbar(label="tau")
    title = "Tau matrix (base)"
    if model_name:
        title = f"{model_name}: " + title
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "tau_matrix.png"))
    plt.close()

def plot_switch1_degradation(switch1_records: list, base_records: list, out_dir: str, model_name: str = None):
    os.makedirs(out_dir, exist_ok=True)
    # build zero-shot baseline per target
    baseline = {}
    for r in base_records:
        if r.get("phase")=="base" and r.get("history_length")==0:
            for m in ("accuracy","exact_match","rouge1","tau"):
                if m in r:
                    baseline[r["task"]] = r[m]
                    break
    # group switch1 by target
    by_target = defaultdict(list)
    for r in switch1_records:
        if r.get("phase")=="switch1" and r.get("order")=="target_recent" and r.get("L_target")==0:
            by_target[r["target"]].append(r)

    for target, recs in by_target.items():
        if target not in baseline:
            continue
        base_val = baseline[target]
        # pick metric
        for m in ("accuracy","exact_match","rouge1","tau"):
            if any(m in r for r in recs):
                metric = m
                break
        else:
            continue
        plt.figure()
        distractors = sorted({r["distractor"] for r in recs})
        for d in distractors:
            series = sorted([r for r in recs if r["distractor"]==d],
                            key=lambda x: x["L_distractor"])
            xs = [r["L_distractor"] for r in series]
            ys = [(r[metric] - base_val)*100.0 for r in series]
            se_key = metric + "_se"
            if all(se_key in r for r in series):
                ses = [r[se_key]*100.0 for r in series]
                lower = [y - s for y, s in zip(ys, ses)]
                upper = [y + s for y, s in zip(ys, ses)]
                plt.fill_between(xs, lower, upper, alpha=0.2)
            plt.plot(xs, ys, marker="o", label=d)
        plt.xlabel("Distractor history length")
        plt.ylabel(f"{metric} % change")
        title = f"{target} — {metric} degradation (switch1)"
        if model_name:
            title = f"{model_name}: " + title
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{target}_switch1_degradation.png"))
        plt.close()

def plot_switch2_degradation(switch2_records: list, base_records: list, out_dir: str, model_name: str = None):
    os.makedirs(out_dir, exist_ok=True)
    # zero-shot baseline per target
    baseline = {}
    for r in base_records:
        if r.get("phase")=="base" and r.get("history_length")==0:
            for m in ("accuracy","exact_match","rouge1","tau"):
                if m in r:
                    baseline[r["task"]] = r[m]
                    break
    # group switch2 by target
    by_target = defaultdict(list)
    for r in switch2_records:
        if r.get("phase")=="switch2" and r.get("order")=="target_recent" and r.get("L_target")==0:
            by_target[r["target"]].append(r)

    for target, recs in by_target.items():
        if target not in baseline:
            continue
        base_val = baseline[target]
        # pick metric
        for m in ("accuracy","exact_match","rouge1","tau"):
            if any(m in r for r in recs):
                metric = m
                break
        else:
            continue
        plt.figure()
        pairs = sorted({f"{r['distractor1']}+{r['distractor2']}" for r in recs})
        for pair in pairs:
            series = sorted([r for r in recs
                             if f"{r['distractor1']}+{r['distractor2']}"==pair],
                            key=lambda x: x["L_distractor"])
            xs = [r["L_distractor"] for r in series]
            ys = [(r[metric] - base_val)*100.0 for r in series]
            se_key = metric + "_se"
            if all(se_key in r for r in series):
                ses = [r[se_key]*100.0 for r in series]
                lower = [y - s for y, s in zip(ys, ses)]
                upper = [y + s for y, s in zip(ys, ses)]
                plt.fill_between(xs, lower, upper, alpha=0.2)
            plt.plot(xs, ys, marker="o", label=pair)
        plt.xlabel("Distractor history length")
        plt.ylabel(f"{metric} % change")
        title = f"{target} — {metric} degradation (switch2)"
        if model_name:
            title = f"{model_name}: " + title
        plt.title(title)
        plt.legend(fontsize="small", ncol=2)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{target}_switch2_degradation.png"))
        plt.close()