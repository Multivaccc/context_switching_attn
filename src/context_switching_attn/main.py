import os
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
    runner = ExperimentRunner(
        # model_name="EleutherAI/pythia-1.4b-deduped",
        model_name = "gpt2",
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
            "dataset": RottenTomatoesDataset(split="test", num_examples=200),
            "type": "clf"
        },
        {
            "name": "gigaword",
            "dataset": GigawordDataset(split="test", num_examples=200),
            "type": "sum"
        },
        {
            "name": "tweetqa",
            "dataset": TweetQADataset(split="test", num_examples=200),
            "type": "qa"
        },
        {
            "name": "code_mbpp",
            "dataset": CodeCompletionDataset(split="test", num_examples=100),
            "type": "code"
        },
        {
            "name": "para_paws",
            "dataset": ParaphraseDataset(split="test", num_examples=200),
            "type": "para"
        },
    ]

    base_recs = runner.run_base(tasks)
    ext_recs  = runner.run_extended(tasks)
    all_recs = base_recs + ext_recs
    save_results(all_recs, "results/all.json", "results/all.csv")
    plot_degradation(all_recs, "results/plots/degradation")
    plot_tau_matrix(all_recs, "results/plots")
    print("Done.")

if __name__ == "__main__":
    main()
