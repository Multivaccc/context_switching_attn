import argparse
import os
import multiprocessing as mp
import torch

from .experiments import run_task_switch_experiments
import json
import pandas as pd
from .utils import (
    save_results,
    plot_base_degradation,
    plot_tau_matrix,
    plot_switch1_degradation,
    plot_switch2_degradation,
)

SUPPORTED_TASKS = [
    "code_completion",
    # "gigaword",
    "mmlu",
    "paraphrase",
    "rotten_tomatoes",
    "tweet_qa",
]

def run_for_model(model_name, history_lengths, num_incontext, eval_size, seed, base_output, blaxkbox):
    model_dir = os.path.join(base_output, model_name)
    os.makedirs(model_dir, exist_ok=True)
    json_path = os.path.join(model_dir, "task_switch.json")
    csv_path = os.path.join(model_dir, "task_switch.csv")

    if os.path.exists(csv_path):
        print(f"Results already exist at {csv_path}, loading results and generating plots...")
        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                records = json.load(f)
        else:
            df = pd.read_csv(csv_path)
            records = df.to_dict(orient="records")
    else:
        records = run_task_switch_experiments(
            model_name=model_name,
            all_tasks=SUPPORTED_TASKS,
            history_lengths=history_lengths,
            num_incontext=num_incontext,
            eval_size=eval_size,
            seed=seed,
            blackbox=blackbox,
        )
        save_results(records, json_path, csv_path)
        print(f"Saved {len(records)} records for {model_name} to {json_path} and {csv_path}")

    # Plotting
    base_records = [r for r in records if r.get("phase") == "base"]
    switch1_records = [r for r in records if r.get("phase") == "switch1"]
    switch2_records = [r for r in records if r.get("phase") == "switch2"]

    # Output directory for plots
    plots_dir = os.path.join(model_dir, "plots")
    # Base degradation
    plot_base_degradation(base_records, plots_dir, model_name)
    # Tau matrix
    plot_tau_matrix(records, plots_dir, model_name)
    # Switch1 degradation
    plot_switch1_degradation(switch1_records, base_records, plots_dir, model_name)
    # Switch2 degradation
    plot_switch2_degradation(switch2_records, base_records, plots_dir, model_name)

def main():
    parser = argparse.ArgumentParser(description="Run and plot task-switching experiments across datasets for multiple models.")
    parser.add_argument("--models", nargs='+', type=str, default=["gpt2"],
        help="List of HuggingFace model names to run (default: gpt2)"
    )
    parser.add_argument("--history_lengths", nargs='+', type=int, default=[0, 1, 2, 3, 4, 5, 6],
        help="List of in-context example counts to sweep"
    )
    parser.add_argument("--num_incontext", type=int, default=5,
        help="Max number of in-context examples per dataset"
    )
    parser.add_argument("--eval_size", type=int, default=None,
        help="Number of evaluation examples (None = default of each dataset)"
    )
    parser.add_argument("--seed", type=int, default=42,
        help="Random seed multiplier (currently unused)"
    )
    parser.add_argument("--output_dir", type=str, default="results",
        help="Base directory to save results"
    )
    parser.add_argument("--blackbox", action="store_true",
        help="Use blackbox model (default: False)"
    )

    args = parser.parse_args()

    for model in args.models:
        run_for_model(
            model_name=model,
            history_lengths=args.history_lengths,
            num_incontext=args.num_incontext,
            eval_size=args.eval_size,
            seed=args.seed,
            base_output=args.output_dir,
            blackbox=args.blackbox,
        )


if __name__ == "__main__":
    main()
