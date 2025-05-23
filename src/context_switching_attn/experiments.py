import math
import numpy as np
import torch
from scipy.stats import norm
from tqdm import tqdm

from .model_wrapper import ModelWrapper
from .metrics import accuracy, exact_match, rouge, meteor
from .prompt_loader import PromptLoader

def wilson_interval(p: float, n: int, z: float = 1.96):
    """Wilson score interval for a proportion p over n trials."""
    if n == 0:
        return 0.0, 0.0
    center = (p + z*z/(2*n)) / (1 + z*z/n)
    half = (z * math.sqrt(p*(1-p)/n + z*z/(4*n*n))) / (1 + z*z/n)
    return center - half, center + half

def sensitivity_slope(xs: list[int], ys: list[float]) -> float:
    """Return slope (dy/dx) of the best-fit line through (xs, ys)."""
    if len(xs) < 2:
        return 0.0
    m, _ = np.polyfit(xs, ys, 1)
    return float(m)

def run_task_switch_experiments(
    model_name: str,
    all_tasks: list[str],
    history_lengths: list[int],
    num_incontext: int,
    eval_size: int | None = None,
    seed: int = 0,
    blackbox: bool = False,
):
    """
    For each target task A and each distractor task B != A, runs:
      - base:     L_target - history_lengths, no B
      - switch1:  L_distractor - history_lengths, no A
      - switch2:  L_target = L_distractor - history_lengths, with both A/B in two orders
    Records accuracy (and CIs) for classification, or exact_match+rouge+meteor for generation,
    then computes a sensitivity slope over history_lengths.
    """
    wrapper = ModelWrapper(model_name, blackbox=blackbox)
    records = []

    for target in tqdm(all_tasks, desc="Target tasks"):
        for distractor in tqdm(all_tasks, desc=f"Distractors for {target}", leave=False):
            if distractor == target:
                continue

            loader = PromptLoader(incontext=distractor, eval=target)
            eval_prompts, references, choices_list = loader.load_eval(eval_size)

            # Base: only target in-context
            base_loader = PromptLoader(incontext=target, eval=target)
            for L_t in tqdm(history_lengths, desc=f"Base L for {target}/{distractor}", leave=False):
                hist_t = base_loader.load_incontext(min(L_t, num_incontext))
                preds = []
                for prompt, ref, choices in zip(eval_prompts, references, choices_list):
                    chat = hist_t + [{"role": "user", "content": prompt}]
                    if choices is None:
                        preds.append(wrapper.generate(chat, pad_token_id=wrapper.tokenizer.pad_token_id))
                    else:
                        pred_idx, _, _ = wrapper.classify(chat, choices)
                        preds.append(pred_idx)

                n = len(preds)
                if choices_list[0] is not None:
                    # classification metrics
                    acc = accuracy(torch.tensor(preds), torch.tensor(references))
                    low, high = wilson_interval(acc, n)
                    records.append({
                        "phase": "base",
                        "target": target,
                        "distractor": None,
                        "order": None,
                        "L_target": L_t,
                        "L_distractor": 0,
                        "accuracy": acc,
                        "accuracy_ci_low": low,
                        "accuracy_ci_high": high,
                    })
                else:
                    # generative metrics
                    em = exact_match(references, preds)
                    rouge_scores = rouge(references, preds)
                    mtr = meteor(references, preds)
                    rec = {
                        "phase": "base",
                        "target": target,
                        "distractor": None,
                        "order": None,
                        "L_target": L_t,
                        "L_distractor": 0,
                        "exact_match": em,
                        "meteor": mtr,
                    }
                    rec.update({f"rouge{k}": v for k, v in rouge_scores.items()})
                    records.append(rec)

            # Switch1: only distractor in-context
            for L_d in tqdm(history_lengths, desc=f"Switch1 L for {target}/{distractor}", leave=False):
                hist_d = loader.load_incontext(L_d)
                preds = []
                for prompt, ref, choices in zip(eval_prompts, references, choices_list):
                    chat = hist_d + [{"role": "user", "content": prompt}]
                    if choices is None:
                        preds.append(wrapper.generate(chat, pad_token_id=wrapper.tokenizer.pad_token_id))
                    else:
                        pred_idx, _, _ = wrapper.classify(chat, choices)
                        preds.append(pred_idx)

                n = len(preds)
                if choices_list[0] is not None:
                    acc = accuracy(torch.tensor(preds), torch.tensor(references))
                    low, high = wilson_interval(acc, n)
                    records.append({
                        "phase": "switch1",
                        "target": target,
                        "distractor": distractor,
                        "order": "distractor_recent",
                        "L_target": 0,
                        "L_distractor": L_d,
                        "accuracy": acc,
                        "accuracy_ci_low": low,
                        "accuracy_ci_high": high,
                    })
                else:
                    em = exact_match(references, preds)
                    rouge_scores = rouge(references, preds)
                    mtr = meteor(references, preds)
                    rec = {
                        "phase": "switch1",
                        "target": target,
                        "distractor": distractor,
                        "order": "distractor_recent",
                        "L_target": 0,
                        "L_distractor": L_d,
                        "exact_match": em,
                        "meteor": mtr,
                    }
                    rec.update({f"rouge{k}": v for k, v in rouge_scores.items()})
                    records.append(rec)

            # Switch2: both in-context, two orders
            for L in tqdm(history_lengths, desc=f"Switch2 L for {target}/{distractor}", leave=False):
                hist_d = loader.load_incontext(L)
                hist_t = PromptLoader(target, target).load_incontext(L)
                for order in ("distractor_recent", "target_recent"):
                    preds = []
                    for prompt, ref, choices in zip(eval_prompts, references, choices_list):
                        if order == "distractor_recent":
                            chat = hist_d + hist_t + [{"role": "user", "content": prompt}]
                        else:
                            chat = hist_t + hist_d + [{"role": "user", "content": prompt}]

                        if choices is None:
                            preds.append(wrapper.generate(chat, pad_token_id=wrapper.tokenizer.pad_token_id))
                        else:
                            pred_idx, _, _ = wrapper.classify(chat, choices)
                            preds.append(pred_idx)

                    n = len(preds)
                    if choices_list[0] is not None:
                        acc = accuracy(torch.tensor(preds), torch.tensor(references))
                        low, high = wilson_interval(acc, n)
                        records.append({
                            "phase": "switch2",
                            "target": target,
                            "distractor": distractor,
                            "order": order,
                            "L_target": L,
                            "L_distractor": L,
                            "accuracy": acc,
                            "accuracy_ci_low": low,
                            "accuracy_ci_high": high,
                        })
                    else:
                        em = exact_match(references, preds)
                        rouge_scores = rouge(references, preds)
                        mtr = meteor(references, preds)
                        rec = {
                            "phase": "switch2",
                            "target": target,
                            "distractor": distractor,
                            "order": order,
                            "L_target": L,
                            "L_distractor": L,
                            "exact_match": em,
                            "meteor": mtr,
                        }
                        rec.update({f"rouge{k}": v for k, v in rouge_scores.items()})
                        records.append(rec)

    # Compute a single sensitivity slope per (phase,target,distractor,order)
    from collections import defaultdict
    grouped = defaultdict(list)
    for r in records:
        key = (r["phase"], r["target"], r.get("distractor"), r.get("order"))
        grouped[key].append(r)

    for key, recs in grouped.items():
        # choose xs = history length (use L_distractor for switch1, L_target otherwise)
        xs = [
            r["L_distractor"] if key[0] == "switch1" else r["L_target"]
            for r in recs
        ]
        # use accuracy or exact_match as y
        if "accuracy" in recs[0]:
            ys = [r["accuracy"] for r in recs]
        else:
            ys = [r["exact_match"] for r in recs]
        slope = sensitivity_slope(xs, ys)
        for r in recs:
            r["sensitivity"] = slope

    return records
