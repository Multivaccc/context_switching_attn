import os
import json
import csv
import pytest
import torch
import matplotlib
matplotlib.use('Agg')  # for headless plotting

import datasets
@pytest.fixture(autouse=True)
def patch_load_dataset(monkeypatch):
    def dummy_load_dataset(name, *args, split=None, trust_remote_code=None, name_arg=None, **kwargs):
        # determine number of examples from split spec e.g. 'test[:N]'
        num = 1
        if isinstance(split, str) and '[' in split:
            import re
            m = re.search(r"\[:(\d+)\]", split)
            if m:
                num = int(m.group(1))
        if name.startswith('cais/mmlu') or name == 'cais/mmlu':
            return [{'question': f'q{i}', 'choices': ['A','B'], 'answer': 'A'} for i in range(num)]
        if name == 'mbpp':
            return [{'text': f't{i}', 'code': f'c{i}'} for i in range(num)]
        if name == 'gigaword':
            return [{'document': f'd{i}', 'summary': f's{i}'} for i in range(num)]
        if name == 'paws':
            return [{'sentence1': f's1_{i}', 'sentence2': f's2_{i}', 'label': 1} for i in range(num)]
        if name == 'rotten_tomatoes':
            return [{'text': f'text{i}', 'label': i % 2} for i in range(num)]
        if name == 'ucsbnlp/tweet_qa':
            return [{'Tweet': f't{i}', 'Question': f'q{i}', 'Answer': [f'a{i}']} for i in range(num)]
        return []
    monkeypatch.setattr(datasets, 'load_dataset', dummy_load_dataset)

# Dataset shape tests
from src.context_switching_attn.dataset.mmlu import MMLUDataset
from src.context_switching_attn.dataset.rotten_tomatoes import RottenTomatoesDataset
from src.context_switching_attn.dataset.gigaword import GigawordDataset
from src.context_switching_attn.dataset.tweet_qa import TweetQADataset
from src.context_switching_attn.dataset.code_completion import CodeCompletionDataset
from src.context_switching_attn.dataset.paraphrase import ParaphraseDataset

@pytest.mark.parametrize("cls,kwargs,keys", [
    (MMLUDataset, {"split":"test","subjects":"abstract_algebra","num_examples":5}, {"prompt","choices","label"}),
    (CodeCompletionDataset, {"split":"test","num_examples":5}, {"prompt","reference"}),
    (GigawordDataset, {"split":"test","num_examples":5}, {"prompt","reference"}),
    (ParaphraseDataset, {"split":"test","num_examples":10}, {"prompt","reference"}),
    (RottenTomatoesDataset, {"split":"test","num_examples":10}, {"prompt","choices","label"}),
    (TweetQADataset, {"split":"test","num_examples":10}, {"prompt","reference"}),
])
def test_dataset_shapes(cls, kwargs, keys):
    ds = cls(**kwargs)
    assert len(ds) == kwargs.get("num_examples", 1)
    item = ds[0]
    assert set(keys).issubset(item.keys())

# Skip heavy ModelWrapper load
@pytest.mark.skip(reason="Skip ModelWrapper in unit tests")
def test_model_wrapper_smoke():
    from src.context_switching_attn.model_wrapper import ModelWrapper
    mw = ModelWrapper("gpt2", device="cpu")
    ids = mw.tokenizer("test", return_tensors="pt").input_ids
    lp = mw.sequence_log_prob(ids)
    assert lp.numel() >= 1

# Metrics tests
from src.context_switching_attn.metrics import accuracy, tau, exact_match, rouge, meteor

def test_accuracy():
    preds = torch.tensor([1,2,3])
    labels = torch.tensor([1,2,0])
    assert accuracy(preds, labels) == pytest.approx(2/3)

def test_tau():
    logp0 = torch.tensor([0.,1.,2.])
    logph = torch.tensor([1.,1.,1.])
    assert tau(logp0, logph) == pytest.approx(((-1)+0+1)/3)

def test_exact_match():
    refs = ["a","b","c"]
    hyps = ["a","wrong","c"]
    assert exact_match(refs, hyps) == pytest.approx(2/3)

def test_rouge_and_meteor():
    refs = ["the cat sat","hello world"]
    hyps = ["the cat sat","hello"]
    r = rouge(refs, hyps)
    assert set(r.keys()) == {"rouge1","rouge2","rougeL"}
    for v in r.values():
        assert 0.0 <= v <= 1.0
    m = meteor(refs, hyps)
    assert m >= 0.0

# Utils tests
from src.context_switching_attn.utils import save_results, plot_base_degradation, plot_tau_matrix, plot_switch1_degradation, plot_switch2_degradation

def test_save_results(tmp_path):
    records = [{"a":1,"b":2},{"b":3,"c":4}]
    j = tmp_path/"out.json"
    c = tmp_path/"out.csv"
    save_results(records, str(j), str(c))
    assert j.exists() and c.exists()
    with open(j) as f:
        assert json.load(f) == records
    with open(c) as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["a"] == "1" and rows[1]["c"] == "4"

@pytest.fixture
def base_and_switch_records():
    base = [{"phase":"base","task":"t1","history_length":0,"accuracy":1.0,"accuracy_se":0.0}]
    sw1  = [{"phase":"switch1","target":"t1","distractor":"d1","order":"target_recent","L_target":0,"L_distractor":0,"accuracy":1.0,"accuracy_se":0.0},
            {"phase":"switch1","target":"t1","distractor":"d1","order":"target_recent","L_target":0,"L_distractor":2,"accuracy":0.8,"accuracy_se":0.1}]
    sw2  = [{"phase":"switch2","target":"t1","distractor1":"d1","distractor2":"d2","order":"target_recent","L_target":0,"L_distractor":0,"accuracy":1.0,"accuracy_se":0.0},
            {"phase":"switch2","target":"t1","distractor1":"d1","distractor2":"d2","order":"target_recent","L_target":0,"L_distractor":2,"accuracy":0.7,"accuracy_se":0.2}]
    return base, sw1, sw2

def test_plot_base_and_tau(tmp_path, base_and_switch_records):
    base, sw1, sw2 = base_and_switch_records
    bd = tmp_path/"plots"/"base"
    plot_base_degradation(base, str(bd))
    assert (bd/"t1_base_degradation.png").exists()
    tau_records=[{"phase":"base","src_task":"t1","tgt_task":"t2","tau":0.5},
                 {"phase":"base","src_task":"t2","tgt_task":"t1","tau":-0.5}]
    plot_tau_matrix(tau_records, str(bd))
    assert (bd/"tau_matrix.png").exists()

def test_plot_switch1(tmp_path, base_and_switch_records):
    base, sw1, _ = base_and_switch_records
    d1 = tmp_path/"plots"/"switch1"
    plot_switch1_degradation(sw1, base, str(d1))
    assert (d1/"t1_switch1_degradation.png").exists()

def test_plot_switch2(tmp_path, base_and_switch_records):
    base, sw1, sw2 = base_and_switch_records
    d2 = tmp_path/"plots"/"switch2"
    plot_switch2_degradation(sw2, base, str(d2))
    assert (d2/"t1_switch2_degradation.png").exists()