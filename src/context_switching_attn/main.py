import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
import tqdm
tqdm.tqdm.monitor_interval = 0

import torch
from dotenv import load_dotenv
from context_switching_attn.dataset.mmlu import MMLUDataset
from context_switching_attn.dataset.rotten_tomatoes import RottenTomatoesDataset
from context_switching_attn.dataset.gigaword import GigawordDataset
from context_switching_attn.dataset.tweet_qa import TweetQADataset
from context_switching_attn.dataset.code_completion import CodeCompletionDataset
from context_switching_attn.dataset.paraphrase import ParaphraseDataset
from context_switching_attn.experiments import ExperimentRunner
from context_switching_attn.utils import save_results, plot_degradation, plot_tau_matrix

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # load_dotenv()
    model_name = "gpt2"
    runner = ExperimentRunner(
        model_name = model_name,
        device=device,
        history_lengths=[0,2,4,6],
        freq_variations=[(1,1),(2,1),(1,2)]
    )

    tasks = [
        {
            "name": "mmlu_aa",
            "dataset": MMLUDataset(split="test", subjects="abstract_algebra", num_examples=50),
            "type": "clf"
        },
        {
            "name": "rotten",
            "dataset": RottenTomatoesDataset(split="test", num_examples=20),
            "type": "clf"
        },
        {
            "name": "gigaword",
            "dataset": GigawordDataset(split="test", num_examples=20),
            "type": "sum"
        },
        {
            "name": "tweetqa",
            "dataset": TweetQADataset(split="test", num_examples=20),
            "type": "qa"
        },
        {
            "name": "code_mbpp",
            "dataset": CodeCompletionDataset(split="test", num_examples=20),
            "type": "code"
        },
        {
            "name": "para_paws",
            "dataset": ParaphraseDataset(split="test", num_examples=20),
            "type": "para"
        },
    ]

    base_recs = runner.run_base(tasks)
    ext_recs  = runner.run_extended(tasks)
    all_recs = base_recs + ext_recs

    out_dir = os.path.join("results", model_name)
    os.makedirs(out_dir, exist_ok=True)
    save_results(all_recs, os.path.join(out_dir, "all.json"), os.path.join(out_dir, "all.csv"))
    plot_degradation(all_recs, os.path.join(out_dir, "plots", "degradation"))
    plot_tau_matrix(all_recs, os.path.join(out_dir, "plots"))
    print("Done.")

if __name__ == "__main__":
    main()
