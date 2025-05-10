import os
import json
import csv
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import defaultdict

sns.set_palette("pastel")
sns.set_style("whitegrid")

def save_results(records: list, json_path: str, csv_path: str):
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(records, f, indent=2)
    if not records:
        return
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
    # Group by target task
    by_task = defaultdict(list)
    for r in base_records:
        if r.get("phase") == "base" and "target" in r:
            by_task[r["target"]].append(r)

    for task, recs in by_task.items():
        # Sort by L_target
        recs = sorted(recs, key=lambda x: x["L_target"])
        # Pick metric in order of preference
        for metric in ("accuracy", "exact_match", "rouge1", "tau"):
            if any(metric in r for r in recs):
                break
        else:
            continue

        xs = [r["L_target"] for r in recs]
        ys = [r[metric] for r in recs]

        plt.figure()
        plt.plot(xs, ys, marker="o", label=metric)

        # confidence‐interval shading if present
        low_key = f"{metric}_ci_low"
        high_key = f"{metric}_ci_high"
        if all(low_key in r and high_key in r for r in recs):
            lows = [r[low_key] for r in recs]
            highs = [r[high_key] for r in recs]
            plt.fill_between(xs, lows, highs, alpha=0.2)

        plt.xlabel("History length")
        plt.ylabel(metric)
        title = f"{task} - {metric} (base)"
        if model_name:
            title = f"{model_name}: {title}"
        plt.title(title)

        # legend on top
        plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=1, frameon=False)
        plt.tight_layout()

        filename = f"{task}_base_degradation.png"
        plt.savefig(os.path.join(out_dir, filename))
        print(f"Saved {task} base degradation plot to {os.path.join(out_dir, filename)}")
        plt.close()

def plot_tau_matrix(records: list, out_dir: str, model_name: str = None):
    os.makedirs(out_dir, exist_ok=True)
    tasks = sorted({r["target"] for r in records if r.get("phase") == "base" and "target" in r})
    if not tasks:
        print("No base records to plot tau matrix, skipping.")
        return

    n = len(tasks)
    mat = np.zeros((n, n))
    for i, task in enumerate(tasks):
        rec = next(
            (r for r in records if r.get("phase") == "base" and r.get("target") == task),
            None,
        )
        if rec:
            mat[i, i] = rec.get("tau", rec.get("sensitivity", 0.0))

    plt.figure(figsize=(6, 6))
    sns.heatmap(
        mat,
        xticklabels=tasks,
        yticklabels=tasks,
        cmap="Pastel1",
        cbar_kws={"label": "tau"},
        square=True,
    )
    plt.xticks(rotation=90)
    title = "Tau matrix (base)"
    if model_name:
        title = f"{model_name}: {title}"
    plt.title(title)
    plt.tight_layout()
    path = os.path.join(out_dir, "tau_matrix.png")
    plt.savefig(path)
    print(f"Saved tau matrix plot to {path}")
    plt.close()

def plot_switch1_degradation(switch1_records: list, base_records: list, out_dir: str, model_name: str = None):
    os.makedirs(out_dir, exist_ok=True)
    # zero‐shot baseline per target
    baseline = {}
    for r in base_records:
        if r.get("phase") == "base" and r.get("L_target") == 0 and "target" in r:
            for m in ("accuracy", "exact_match", "rouge1", "tau"):
                if m in r:
                    baseline[r["target"]] = r[m]
                    break

    # group switch1 by target
    by_target = defaultdict(list)
    for r in switch1_records:
        if r.get("phase") == "switch1" and r.get("L_target") == 0 and "target" in r:
            by_target[r["target"]].append(r)

    for target, recs in by_target.items():
        if target not in baseline:
            continue
        base_val = baseline[target]
        for metric in ("accuracy", "exact_match", "rouge1", "tau"):
            if any(metric in r for r in recs):
                break
        else:
            continue

        plt.figure()
        for d in sorted({r["distractor"] for r in recs if "distractor" in r}):
            series = sorted(
                [r for r in recs if r.get("distractor") == d],
                key=lambda x: x["L_distractor"],
            )
            xs = [r["L_distractor"] for r in series]
            ys = [(r[metric] - base_val) * 100.0 for r in series]
            plt.plot(xs, ys, marker="o", label=d)

            # CI shading
            low_key, high_key = f"{metric}_ci_low", f"{metric}_ci_high"
            if all(low_key in r and high_key in r for r in series):
                lows = [(r[low_key] - base_val) * 100.0 for r in series]
                highs = [(r[high_key] - base_val) * 100.0 for r in series]
                plt.fill_between(xs, lows, highs, alpha=0.2)

        plt.xlabel("Distractor history length")
        plt.ylabel(f"{metric} % change")
        title = f"{target} - {metric} degradation (switch1)"
        if model_name:
            title = f"{model_name}: {title}"
        plt.title(title)
        plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3, frameon=False)
        plt.tight_layout()

        filename = f"{target}_switch1_degradation.png"
        plt.savefig(os.path.join(out_dir, filename))
        print(f"Saved {target} switch1 degradation plot to {os.path.join(out_dir, filename)}")
        plt.close()

def plot_switch2_degradation(switch2_records: list, base_records: list, out_dir: str, model_name: str = None):
    os.makedirs(out_dir, exist_ok=True)
    # zero‐shot baseline per target
    baseline = {}
    for r in base_records:
        if r.get("phase") == "base" and r.get("L_target") == 0 and "target" in r:
            for m in ("accuracy", "exact_match", "rouge1", "tau"):
                if m in r:
                    baseline[r["target"]] = r[m]
                    break

    # group switch2 by target
    by_target = defaultdict(list)
    for r in switch2_records:
        if (
            r.get("phase") == "switch2"
            and r.get("order") == "target_recent"
            and r.get("L_target") == 0
            and "target" in r
        ):
            by_target[r["target"]].append(r)

    for target, recs in by_target.items():
        if target not in baseline:
            continue
        base_val = baseline[target]
        for metric in ("accuracy", "exact_match", "rouge1", "tau"):
            if any(metric in r for r in recs):
                break
        else:
            continue

        plt.figure()
        for distractor, order in sorted({(r["distractor"], r["order"]) for r in recs}):
            series = sorted(
                [r for r in recs if r.get("distractor") == distractor and r.get("order") == order],
                key=lambda x: x["L_distractor"],
            )
            xs = [r["L_distractor"] for r in series]
            ys = [(r[metric] - base_val) * 100.0 for r in series]
            plt.plot(xs, ys, marker="o", label=f"{distractor} ({order})")

            low_key, high_key = f"{metric}_ci_low", f"{metric}_ci_high"
            if all(low_key in r and high_key in r for r in series):
                lows = [(r[low_key] - base_val) * 100.0 for r in series]
                highs = [(r[high_key] - base_val) * 100.0 for r in series]
                plt.fill_between(xs, lows, highs, alpha=0.2)

        plt.xlabel("Distractor history length")
        plt.ylabel(f"{metric} % change")
        title = f"{target} - {metric} degradation (switch2)"
        if model_name:
            title = f"{model_name}: {title}"
        plt.title(title)
        plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3, frameon=False)
        plt.tight_layout()

        filename = f"{target}_switch2_degradation.png"
        plt.savefig(os.path.join(out_dir, filename))
        print(f"Saved {target} switch2 degradation plot to {os.path.join(out_dir, filename)}")
        plt.close()
