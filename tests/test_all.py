import os
import json
import pytest
import torch

from context_switching_attn.model_wrapper import ModelWrapper
from context_switching_attn.metrics import accuracy, exact_match, tau as tau_fn
from context_switching_attn.experiments import wilson_interval, sensitivity_slope
from context_switching_attn.utils import (
    save_results,
    plot_base_degradation,
    plot_switch1_degradation,
    plot_switch2_degradation,
    plot_tau_matrix,
)

def test_classify_probs_sum_to_one():
    mw = ModelWrapper("gpt2")
    history = [{"role": "user", "content": "Hello"}]
    choices = ["Yes", "No", "Maybe"]
    pred, conf, probs = mw.classify(history, choices)
    assert abs(sum(probs) - 1.0) < 1e-4
    assert 0 <= pred < len(choices)
    assert 0.0 <= conf <= 1.0

def test_generate_returns_nonempty_string():
    mw = ModelWrapper("gpt2")
    history = [{"role": "user", "content": "Once upon a time"}]
    out = mw.generate(history, max_new_tokens=5)
    assert isinstance(out, str)
    assert len(out) > 0

def test_metrics_basic_behaviour():
    # accuracy
    preds = torch.tensor([1, 2, 2, 1])
    labels = torch.tensor([1, 2, 0, 1])
    assert abs(accuracy(preds, labels) - 0.75) < 1e-6

    # exact_match
    refs = ["a", "b", "c"]
    hyps = ["a", "x", "c"]
    assert abs(exact_match(refs, hyps) - (2 / 3)) < 1e-6

    # tau
    logp0 = torch.tensor([0.0, 1.0, 2.0])
    logph = torch.tensor([0.0, 0.5, 1.5])
    expected = ((0.0 - 0.0) + (1.0 - 0.5) + (2.0 - 1.5)) / 3
    assert abs(tau_fn(logp0, logph) - expected) < 1e-6

    # wilson_interval
    low0, high0 = wilson_interval(0.5, 0)
    assert low0 == 0.0 and high0 == 0.0
    low1, high1 = wilson_interval(0.5, 10)
    assert 0.0 <= low1 <= high1 <= 1.0

    # sensitivity_slope
    assert sensitivity_slope([1], [2]) == 0.0
    assert abs(sensitivity_slope([0, 1], [0, 2]) - 2.0) < 1e-6

def test_save_results_writes_json_and_csv(tmp_path):
    recs = [
        {"a": 1, "b": 2},
        {"a": 3, "c": 4},
    ]
    jpath = tmp_path / "out.json"
    cpath = tmp_path / "out.csv"
    save_results(recs, str(jpath), str(cpath))

    # JSON contents
    loaded = json.loads(jpath.read_text())
    assert loaded == recs

    # CSV header + rows
    lines = cpath.read_text().splitlines()
    header = lines[0].split(",")
    # union of keys in insertion order: a, b, c
    assert header == ["a", "b", "c"]
    row1 = lines[1].split(",")
    assert row1 == ["1", "2", ""]
    row2 = lines[2].split(",")
    assert row2 == ["3", "", "4"]

def make_dummy_records():
    recs = []
    # base
    for t in ("A", "B"):
        for L in (0, 1, 2):
            acc = 0.5 + 0.1 * L
            low = max(0.0, acc - 0.05)
            high = min(1.0, acc + 0.05)
            recs.append({
                "phase": "base",
                "target": t,
                "L_target": L,
                "accuracy": acc,
                "accuracy_ci_low": low,
                "accuracy_ci_high": high,
            })
    # switch1
    for t in ("A", "B"):
        for Ld in (0, 1, 2):
            acc = 0.4 + 0.05 * Ld
            low = max(0.0, acc - 0.05)
            high = min(1.0, acc + 0.05)
            recs.append({
                "phase": "switch1",
                "target": t,
                "distractor": "X",
                "order": "distractor_recent",
                "L_target": 0,
                "L_distractor": Ld,
                "accuracy": acc,
                "accuracy_ci_low": low,
                "accuracy_ci_high": high,
            })
    # switch2
    for t in ("A", "B"):
        for Ld in (0, 1, 2):
            acc = 0.6 - 0.02 * Ld
            low = max(0.0, acc - 0.05)
            high = min(1.0, acc + 0.05)
            recs.append({
                "phase": "switch2",
                "target": t,
                "distractor": "X",
                "order": "target_recent",
                "L_target": 0,
                "L_distractor": Ld,
                "accuracy": acc,
                "accuracy_ci_low": low,
                "accuracy_ci_high": high,
            })
    return recs

@pytest.mark.parametrize("func, args", [
    (plot_base_degradation, ([r for r in make_dummy_records() if r["phase"] == "base"],)),
    (plot_switch1_degradation, (
        [r for r in make_dummy_records() if r["phase"] == "switch1"],
        [r for r in make_dummy_records() if r["phase"] == "base"],
    )),
    (plot_switch2_degradation, (
        [r for r in make_dummy_records() if r["phase"] == "switch2"],
        [r for r in make_dummy_records() if r["phase"] == "base"],
    )),
    (plot_tau_matrix, ([r for r in make_dummy_records()],)),
])

def test_plot_functions_do_not_crash_and_create_files(tmp_path, func, args):
    outdir = tmp_path / "plots"
    os.makedirs(outdir, exist_ok=True)
    func(*args, str(outdir), model_name="test")
    files = os.listdir(str(outdir))
    assert any(f.endswith(".png") for f in files)

def test_plot_functions_empty_inputs(tmp_path, capsys):
    outdir = tmp_path / "plots_empty"
    os.makedirs(outdir, exist_ok=True)
    plot_base_degradation([], str(outdir))
    plot_switch1_degradation([], [], str(outdir))
    plot_switch2_degradation([], [], str(outdir))
    plot_tau_matrix([], str(outdir))
    captured = capsys.readouterr()
    assert "skipping" in captured.out.lower()
