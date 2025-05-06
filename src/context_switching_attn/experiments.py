import torch
import math
from itertools import combinations
from context_switching_attn.model_wrapper import ModelWrapper
from context_switching_attn.metrics import tau, exact_match, rouge, meteor
from tqdm import tqdm

class ExperimentRunner:
    def __init__(self, model_name: str, device: str = "cpu", history_lengths=[0, 2, 4, 6]):
        self.device = device
        self.model = ModelWrapper(model_name, device)
        self.H = history_lengths  # history lengths for both target and distractor
        self.orders = ["target_recent", "distr_recent"]  # two orders: target most recent, or distractor most recent

    def run_base(self, tasks):
        records = []
        raw_lists = {t["name"]: list(t["dataset"]) for t in tasks}
        with torch.inference_mode():
            for t in tqdm(tasks, desc="Tasks"):
                examples = raw_lists[t["name"]]
                n_ex = len(examples)
                for L in tqdm(self.H, desc="History Lengths", leave=False):
                    hist_prompts = [ex["prompt"] for ex in examples[:L]]
                    # classification
                    if t["type"] == "clf":
                        logp0, logph, acc_mean, acc_se = self._eval_clf(examples, hist_prompts)
                        delta = logp0 - logph
                        tau_mean = delta.mean().item()
                        tau_std = delta.std(unbiased=False).item()
                        tau_se = tau_std / math.sqrt(delta.numel())
                        rec = {
                            "phase": "base",
                            "task": t["name"],
                            "history_length": L,
                            "accuracy": acc_mean,
                            "accuracy_se": acc_se,
                            "tau": tau_mean,
                            "tau_se": tau_se,
                        }
                    # generation / other
                    else:
                        logp0, logph, metrics = self._eval_gen(examples, hist_prompts, t["type"])
                        delta = logp0 - logph
                        tau_mean = delta.mean().item()
                        tau_std = delta.std(unbiased=False).item()
                        tau_se = tau_std / math.sqrt(delta.numel())
                        rec = {
                            "phase": "base",
                            "task": t["name"],
                            "history_length": L,
                            "tau": tau_mean,
                            "tau_se": tau_se,
                        }
                        rec.update(metrics)
                    records.append(rec)

        # cross-task tau matrix
        combs = list(combinations(tasks, 2))
        for t1, t2 in tqdm(combs, desc="Cross-task pairs"):
            for L in tqdm(self.H, desc="History Lengths", leave=False):
                hist = [ex["prompt"] for ex in raw_lists[t1["name"]][:L]]
                ex0 = raw_lists[t2["name"]][0:1]
                # eval on a single example
                if t2["type"] == "clf":
                    logp0, _, _, _ = self._eval_clf(ex0, [])
                    _, logph, _, _ = self._eval_clf(ex0, hist)
                else:
                    logp0, _, _ = self._eval_gen(ex0, [], t2["type"])
                    _, logph, _ = self._eval_gen(ex0, hist, t2["type"])

                tau_val = tau(logp0, logph)
                records.append({
                    "phase": "base",
                    "src_task": t1["name"],
                    "tgt_task": t2["name"],
                    "history_length": L,
                    "tau": tau_val
                })
                records.append({
                    "phase": "base",
                    "src_task": t2["name"],
                    "tgt_task": t1["name"],
                    "history_length": L,
                    "tau": -tau_val
                })

        return records

    def _eval_clf(self, examples, history_prompts):
        """
        Returns: logp0 tensor, logph tensor, accuracy mean, accuracy se
        """
        logp0_list = []
        logph_list = []
        accs = []
        for ex in tqdm(examples, desc="Examples", leave=False):
            prompt, choices, label = ex["prompt"], ex["choices"], ex["label"]
            lps0 = self._choice_logps("", prompt, choices)
            lpsH = self._choice_logps("".join(history_prompts), prompt, choices)
            logp0_list.append(lps0[label].item())
            logph_list.append(lpsH[label].item())
            accs.append(int(torch.argmax(lpsH).item() == label))

        logp0 = torch.tensor(logp0_list)
        logph = torch.tensor(logph_list)
        accs_t = torch.tensor(accs, dtype=torch.float32)
        mean = accs_t.mean().item()
        std = accs_t.std(unbiased=False).item()
        se = std / math.sqrt(accs_t.numel())
        return logp0, logph, mean, se

    def _choice_logps(self, hist, prompt, choices):
        logps = []
        for ch in choices:
            seq = hist + prompt + ch
            ids = self.model.tokenizer(seq, return_tensors="pt", padding=True).input_ids.to(self.device)
            lp = self.model.sequence_log_prob(ids)[0]
            logps.append(lp)
        return torch.tensor(logps)

    def _eval_gen(self, examples, history_prompts, task_type):
        refs, hyps, logp0_list, logph_list = [], [], [], []
        max_new = 50
        batch_size = 8
        for i in range(0, len(examples), batch_size):
            batch = examples[i:i + batch_size]
            prompts = [ex["prompt"] for ex in batch]
            refs_batch = [ex["reference"] for ex in batch]
            # log-prob no history
            input0 = [p + r for p, r in zip(prompts, refs_batch)]
            ids0 = self.model.tokenizer(input0, return_tensors="pt", padding=True).input_ids.to(self.device)
            lp0 = self.model.sequence_log_prob(ids0)
            # log-prob with history
            hist_str = "".join(history_prompts)
            inputH = [hist_str + p + r for p, r in zip(prompts, refs_batch)]
            idsH = self.model.tokenizer(inputH, return_tensors="pt", padding=True).input_ids.to(self.device)
            lpH = self.model.sequence_log_prob(idsH)
            # generation
            gen_prompts = [hist_str + p for p in prompts]
            hyps_batch = self.model.generate_greedy(gen_prompts, max_new_tokens=max_new)

            refs.extend(refs_batch)
            hyps.extend(hyps_batch)
            logp0_list.extend(lp0.cpu().tolist())
            logph_list.extend(lpH.cpu().tolist())

        logp0 = torch.tensor(logp0_list)
        logph = torch.tensor(logph_list)
        # pick metrics
        if task_type in ("code", "qa"):
            m_val = exact_match(refs, hyps)
            metrics = {"exact_match": m_val}
        else:
            rouge_dict = rouge(refs, hyps)
            meteor_val = meteor(refs, hyps)
            metrics = {
                "rouge1": rouge_dict["rouge1"],
                "rouge2": rouge_dict["rouge2"],
                "rougeL": rouge_dict["rougeL"],
                "meteor": meteor_val
            }

        return logp0, logph, metrics

    def run_task_switching(self, tasks):
        recs = []
        raw = {t["name"]: list(t["dataset"]) for t in tasks}
        for target in tqdm(tasks, desc="Switch1: single distractor"):
            tgt = target["name"]
            examples = raw[tgt]
            for distractor in tasks:
                dst = distractor["name"]
                if dst == tgt:
                    continue
                for L_t in self.H:
                    hist_t = [ex["prompt"] for ex in examples[:L_t]]
                    for L_d in self.H:
                        hist_d = [ex["prompt"] for ex in raw[dst][:L_d]]
                        for order in self.orders:
                            hist = (hist_d + hist_t) if order == "target_recent" else (hist_t + hist_d)

                            if target["type"] == "clf":
                                logp0, logph, acc_mean, acc_se = self._eval_clf(examples, hist)
                                delta = logp0 - logph
                                tau_mean = delta.mean().item()
                                tau_std = delta.std(unbiased=False).item()
                                tau_se = tau_std / math.sqrt(delta.numel())
                                rec = {
                                    "phase": "switch1",
                                    "target": tgt,
                                    "distractor": dst,
                                    "L_target": L_t,
                                    "L_distractor": L_d,
                                    "order": order,
                                    "accuracy": acc_mean,
                                    "accuracy_se": acc_se,
                                    "tau": tau_mean,
                                    "tau_se": tau_se,
                                }
                            else:
                                logp0, logph, metrics = self._eval_gen(examples, hist, target["type"])
                                delta = logp0 - logph
                                tau_mean = delta.mean().item()
                                tau_std = delta.std(unbiased=False).item()
                                tau_se = tau_std / math.sqrt(delta.numel())
                                rec = {
                                    "phase": "switch1",
                                    "target": tgt,
                                    "distractor": dst,
                                    "L_target": L_t,
                                    "L_distractor": L_d,
                                    "order": order,
                                    "tau": tau_mean,
                                    "tau_se": tau_se,
                                }
                                rec.update(metrics)
                            recs.append(rec)
        return recs

    def run_two_distractor_switch(self, tasks):
        recs = []
        raw = {t["name"]: list(t["dataset"]) for t in tasks}
        for target in tqdm(tasks, desc="Switch2: two distractors"):
            tgt = target["name"]
            examples = raw[tgt]
            others = [t for t in tasks if t["name"] != tgt]
            for d1, d2 in combinations(others, 2):
                n1, n2 = d1["name"], d2["name"]
                for L_t in self.H:
                    hist_t = [ex["prompt"] for ex in examples[:L_t]]
                    for L_d in self.H:
                        hist_d1 = [ex["prompt"] for ex in raw[n1][:L_d]]
                        hist_d2 = [ex["prompt"] for ex in raw[n2][:L_d]]
                        hist_d = hist_d1 + hist_d2
                        for order in self.orders:
                            hist = (hist_d + hist_t) if order == "target_recent" else (hist_t + hist_d)

                            if target["type"] == "clf":
                                logp0, logph, acc_mean, acc_se = self._eval_clf(examples, hist)
                                delta = logp0 - logph
                                tau_mean = delta.mean().item()
                                tau_std = delta.std(unbiased=False).item()
                                tau_se = tau_std / math.sqrt(delta.numel())
                                rec = {
                                    "phase": "switch2",
                                    "target": tgt,
                                    "distractor1": n1,
                                    "distractor2": n2,
                                    "L_target": L_t,
                                    "L_distractor": L_d,
                                    "order": order,
                                    "accuracy": acc_mean,
                                    "accuracy_se": acc_se,
                                    "tau": tau_mean,
                                    "tau_se": tau_se,
                                }
                            else:
                                logp0, logph, metrics = self._eval_gen(examples, hist, target["type"])
                                delta = logp0 - logph
                                tau_mean = delta.mean().item()
                                tau_std = delta.std(unbiased=False).item()
                                tau_se = tau_std / math.sqrt(delta.numel())
                                rec = {
                                    "phase": "switch2",
                                    "target": tgt,
                                    "distractor1": n1,
                                    "distractor2": n2,
                                    "L_target": L_t,
                                    "L_distractor": L_d,
                                    "order": order,
                                    "tau": tau_mean,
                                    "tau_se": tau_se,
                                }
                                rec.update(metrics)
                            recs.append(rec)
        return recs
