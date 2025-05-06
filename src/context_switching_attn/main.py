import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
import tqdm
tqdm.tqdm.monitor_interval = 0

import torch
from context_switching_attn.dataset.mmlu import MMLUDataset
from context_switching_attn.dataset.rotten_tomatoes import RottenTomatoesDataset
from context_switching_attn.dataset.gigaword import GigawordDataset
from context_switching_attn.dataset.tweet_qa import TweetQADataset
from context_switching_attn.dataset.code_completion import CodeCompletionDataset
from context_switching_attn.dataset.paraphrase import ParaphraseDataset
from context_switching_attn.experiments import ExperimentRunner
from context_switching_attn.utils import (
    save_results,
    plot_base_degradation,
    plot_tau_matrix,
    plot_switch1_degradation,
    plot_switch2_degradation,
)

def main():
    # device setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_name = 'gpt2'

    # runner initialization
    runner = ExperimentRunner(
        model_name=model_name,
        device=device,
        history_lengths=[0, 2, 4, 6]
    )

    # define tasks
    tasks = [
        {
            'name': 'mmlu_aa',
            'dataset': MMLUDataset(split='test', subjects='abstract_algebra', num_examples=50),
            'type': 'clf'
        },
        {
            'name': 'rotten',
            'dataset': RottenTomatoesDataset(split='test', num_examples=20),
            'type': 'clf'
        },
        {
            'name': 'gigaword',
            'dataset': GigawordDataset(split='test', num_examples=20),
            'type': 'sum'
        },
        {
            'name': 'tweetqa',
            'dataset': TweetQADataset(split='test', num_examples=20),
            'type': 'qa'
        },
        {
            'name': 'code_mbpp',
            'dataset': CodeCompletionDataset(split='test', num_examples=20),
            'type': 'code'
        },
        {
            'name': 'para_paws',
            'dataset': ParaphraseDataset(split='test', num_examples=20),
            'type': 'para'
        },
    ]

    # run experiments
    base_recs = runner.run_base(tasks)
    switch1_recs = runner.run_task_switching(tasks)
    switch2_recs = runner.run_two_distractor_switch(tasks)

    all_recs = base_recs + switch1_recs + switch2_recs

    # save results
    out_dir = os.path.join('results', model_name)
    os.makedirs(out_dir, exist_ok=True)
    save_results(
        all_recs,
        os.path.join(out_dir, 'all.json'),
        os.path.join(out_dir, 'all.csv')
    )

    # split for plotting
    per_task_records = [r for r in all_recs if r.get("phase")=="base" and "task" in r]
    cross_task_records = [r for r in all_recs if r.get("phase")=="base" and "src_task" in r]

    plot_base_degradation(
        per_task_records,
        os.path.join(out_dir, 'plots', 'base'),
        model_name=model_name
    )

    plot_tau_matrix(
        cross_task_records,
        os.path.join(out_dir, 'plots', 'base'),
        model_name=model_name
    )

    plot_switch1_degradation(
        switch1_recs,
        base_recs,
        os.path.join(out_dir, 'plots', 'switch1'),
        model_name=model_name
    )

    plot_switch2_degradation(
        switch2_recs,
        base_recs,
        os.path.join(out_dir, 'plots', 'switch2'),
        model_name=model_name
    )

    print('Done.')

if __name__ == '__main__':
    main()