import torch
from itertools import combinations
from context_switching_attn.model_wrapper import ModelWrapper
from context_switching_attn.metrics import accuracy, tau, rouge, meteor, exact_match
from tqdm import tqdm

class ExperimentRunner:
    def __init__(self, model_name: str, device: str="cpu",
                 history_lengths=[0,3,6], freq_variations=[(1,1)]):
        self.device = device
        self.model = ModelWrapper(model_name, device)
        self.H = history_lengths
        self.freqs = freq_variations

    def run_base(self, tasks):
        records = []
        raw_lists = { t["name"]: list(t["dataset"]) for t in tasks }
        with torch.inference_mode():
            for t in tqdm(tasks, desc="Tasks"):
                examples = raw_lists[t["name"]]
                for L in tqdm(self.H, desc="History Lengths", leave=False):
                    hist_prompts = [ex["prompt"] for ex in examples[:L]]
                    if t["type"] == "clf":
                        logp0, logph, acc = self._eval_clf(examples, hist_prompts)
                        rec = {
                            "phase": "base",
                            "task": t["name"],
                            "history_length": L,
                            "accuracy": acc,
                            "tau": tau(logp0, logph)
                        }
                    else:
                        logp0, logph, metrics = self._eval_gen(examples, hist_prompts, t["type"])
                        rec = {
                            "phase": "base",
                            "task": t["name"],
                            "history_length": L,
                            "tau": tau(logp0, logph)
                        }
                        rec.update(metrics)
                    records.append(rec)
        combs = list(combinations(tasks, 2))
        for t1, t2 in tqdm(combs, desc="Cross-task pairs"):
            for L in tqdm(self.H, desc="History Lengths", leave=False):
                hist = [ex["prompt"] for ex in raw_lists[t1["name"]][:L]]
                ex = raw_lists[t2["name"]][0]
                if t2["type"] == "clf":
                    logp0, _, _ = self._eval_clf([ex], [])
                    _, logph, _ = self._eval_clf([ex], hist)
                else:
                    logp0, _, _ = self._eval_gen([ex], [], t2["type"])
                    _, logph, _ = self._eval_gen([ex], hist, t2["type"])
                records.append({
                    "phase": "base",
                    "src_task": t1["name"],
                    "tgt_task": t2["name"],
                    "history_length": L,
                    "tau": tau(logp0, logph)
                })
                records.append({
                    "phase": "base",
                    "src_task": t2["name"],
                    "tgt_task": t1["name"],
                    "history_length": L,
                    "tau": -tau(logp0, logph)
                })
        return records

    def _eval_clf(self, examples, history_prompts):
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
        return (
            torch.tensor(logp0_list),
            torch.tensor(logph_list),
            torch.tensor(accs).float().mean().item()
        )

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
            batch = examples[i:i+batch_size]
            prompts = [ex["prompt"] for ex in batch]
            refs_batch = [ex["reference"] for ex in batch]
            input_texts0 = [p + r for p, r in zip(prompts, refs_batch)]
            ids0 = self.model.tokenizer(input_texts0, return_tensors="pt", padding=True).input_ids.to(self.device)
            lp0 = self.model.sequence_log_prob(ids0)
            hist_str = "".join(history_prompts)
            input_textsH = [hist_str + p + r for p, r in zip(prompts, refs_batch)]
            idsH = self.model.tokenizer(input_textsH, return_tensors="pt", padding=True).input_ids.to(self.device)
            lpH = self.model.sequence_log_prob(idsH)
            gen_prompts = [hist_str + p for p in prompts]
            hyps_batch = self.model.generate_greedy(gen_prompts, max_new_tokens=max_new)
            refs.extend(refs_batch)
            hyps.extend(hyps_batch)
            logp0_list.extend(lp0.cpu())
            logph_list.extend(lpH.cpu())
        logp0 = torch.stack([torch.tensor(x) for x in logp0_list])
        logph = torch.stack([torch.tensor(x) for x in logph_list])
        if task_type == "code":
            m = exact_match(refs, hyps)
            metrics = {"exact_match": m}
        elif task_type == "qa":
            m = exact_match(refs, hyps)
            metrics = {"exact_match": m}
        else:
            rs = rouge(refs, hyps)
            ms = meteor(refs, hyps)
            metrics = {
                "rouge1": rs["rouge1"],
                "rouge2": rs["rouge2"],
                "rougeL": rs["rougeL"],
                "meteor": ms
            }
        return logp0, logph, metrics

    def run_extended(self, tasks):
        recs = []
        raw_lists = { t["name"]: list(t["dataset"]) for t in tasks }
        combs = list(combinations(tasks, 2))
        for t1, t2 in tqdm(combs, desc="Extended pairs"):
            for (a, b) in tqdm(self.freqs, desc="Mix ratios", leave=False):
                for L in tqdm(self.H, desc="History Lengths", leave=False):
                    ex1 = raw_lists[t1["name"]][:L]
                    ex2 = raw_lists[t2["name"]][:L]
                    if not ex1:
                        continue
                    hist = (
                        [e["prompt"] for _ in range(a) for e in ex1] +
                        [e["prompt"] for _ in range(b) for e in ex2]
                    )
                    if t1["type"] == "clf":
                        logp0, _, acc0 = self._eval_clf(ex1, [])
                        _, logph, accH = self._eval_clf(ex1, hist)
                        metrics = {"accuracy": accH}
                    else:
                        logp0, _, _ = self._eval_gen(ex1, [], t1["type"])
                        _, logph, metrics = self._eval_gen(ex1, hist, t1["type"])
                    recs.append({
                        "phase": "extended",
                        "pair": f"{t1['name']}+{t2['name']}",
                        "mix": f"{a}:{b}",
                        "history_length": L,
                        "tau": tau(logp0, logph),
                        **metrics
                    })
        return recs
