import os
import json
import csv
import pytest
import torch
from src.context_switching_attn.metrics import accuracy, tau, exact_match, rouge, meteor
from src.context_switching_attn.utils import save_results, plot_base_degradation, plot_tau_matrix, plot_switch1_degradation, plot_switch2_degradation

def test_accuracy():
    preds = torch.tensor([1, 2, 3])
    labels = torch.tensor([1, 2, 0])
    assert accuracy(preds, labels) == pytest.approx(2/3)

def test_tau():
    logp0 = torch.tensor([0.0, 1.0, 2.0])
    logph = torch.tensor([1.0, 1.0, 1.0])
    assert tau(logp0, logph) == pytest.approx(((-1) + 0 + 1) / 3)

def test_exact_match():
    refs = ["a", "b", "c"]
    hyps = ["a", "wrong", "c"]
    assert exact_match(refs, hyps) == pytest.approx(2/3)

def test_rouge_and_meteor():
    refs = ["the cat sat", "hello world"]
    hyps = ["the cat sat", "hello"]
    r = rouge(refs, hyps)
    assert set(r.keys()) == {"rouge1", "rouge2", "rougeL"}
    for v in r.values():
        assert 0.0 <= v <= 1.0
    m = meteor(refs, hyps)
    assert m >= 0.0

def test_save_results(tmp_path):
    records = [{"a": 1, "b": 2}, {"b": 3, "c": 4}]
    json_path = tmp_path / "test.json"
    csv_path = tmp_path / "test.csv"
    save_results(records, str(json_path), str(csv_path))
    assert json_path.exists()
    assert csv_path.exists()
    # validate JSON content
    with open(json_path) as f:
        assert json.load(f) == records
    # validate CSV header and rows
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert rows[0]["a"] == "1"
    assert rows[1]["c"] == "4"

def test_plot_base_and_tau(tmp_path):
    # base degradation plot
    base_records = [
        {"phase": "base", "task": "t1", "history_length": 0, "accuracy": 1.0, "accuracy_se": 0.0},
        {"phase": "base", "task": "t1", "history_length": 2, "accuracy": 0.5, "accuracy_se": 0.1},
    ]
    base_dir = tmp_path / "plots" / "base"
    plot_base_degradation(base_records, str(base_dir))
    assert (base_dir / "t1_base_degradation.png").exists()

    # tau matrix plot
    tau_records = [
        {"phase": "base", "src_task": "t1", "tgt_task": "t2", "tau": 0.2},
        {"phase": "base", "src_task": "t2", "tgt_task": "t1", "tau": -0.2},
    ]
    tau_dir = tmp_path / "plots" / "base"
    plot_tau_matrix(tau_records, str(tau_dir))
    assert (tau_dir / "tau_matrix.png").exists()

def test_plot_switch1(tmp_path):
    base_records = [
        {"phase": "base", "task": "t1", "history_length": 0, "accuracy": 1.0},
    ]
    switch1_records = [
        {"phase": "switch1", "target": "t1", "distractor": "d1", "order": "target_recent", "L_target": 0, "L_distractor": 0, "accuracy": 1.0, "accuracy_se": 0.0},
        {"phase": "switch1", "target": "t1", "distractor": "d1", "order": "target_recent", "L_target": 0, "L_distractor": 2, "accuracy": 0.8, "accuracy_se": 0.1},
    ]
    dir1 = tmp_path / "plots" / "switch1"
    plot_switch1_degradation(switch1_records, base_records, str(dir1))
    assert (dir1 / "t1_switch1_degradation.png").exists()

def test_plot_switch2(tmp_path):
    base_records = [
        {"phase": "base", "task": "t1", "history_length": 0, "accuracy": 1.0},
    ]
    switch2_records = [
        {"phase": "switch2", "target": "t1", "distractor1": "d1", "distractor2": "d2", "order": "target_recent", "L_target": 0, "L_distractor": 0, "accuracy": 1.0, "accuracy_se": 0.0},
        {"phase": "switch2", "target": "t1", "distractor1": "d1", "distractor2": "d2", "order": "target_recent", "L_target": 0, "L_distractor": 2, "accuracy": 0.7, "accuracy_se": 0.2},
    ]
    dir2 = tmp_path / "plots" / "switch2"
    plot_switch2_degradation(switch2_records, base_records, str(dir2))
    assert (dir2 / "t1_switch2_degradation.png").exists()
