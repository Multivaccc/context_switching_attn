import torch
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
import nltk

try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet", quiet=True)

def accuracy(preds: torch.Tensor, labels: torch.Tensor):
    return (preds == labels).float().mean().item()

def format_error_rate(preds: list, labels: list):
    errs = sum(1 for p in preds if p is None)
    return errs / len(preds)

def tau(logp0: torch.Tensor, logph: torch.Tensor):
    return (logp0 - logph).mean().item()

def rouge(refs: list, hyps: list):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    agg = {k: 0.0 for k in ["rouge1", "rouge2", "rougeL"]}

    for ref, hyp in zip(refs, hyps):
        if isinstance(ref, list):
            ref_text = ref[0] if ref else ""
        else:
            ref_text = ref or ""
        scores = scorer.score(ref_text, hyp)
        for k in agg:
            agg[k] += scores[k].fmeasure

    return {k: agg[k] / len(refs) if refs else 0.0 for k in agg}

def meteor(refs: list, hyps: list):
    total = 0.0
    for ref, hyp in zip(refs, hyps):
        if isinstance(ref, list):
            ref_text = ref[0] if ref else ""
        else:
            ref_text = ref or ""
        ref_tokens = ref_text.split()
        hyp_tokens = hyp.split()
        total += meteor_score([ref_tokens], hyp_tokens)

    return total / len(refs) if refs else 0.0

def exact_match(refs: list, hyps: list):
    matches = 0
    total = 0
    for r, h in zip(refs, hyps):
        h_text = h.strip()
        total += 1
        if isinstance(r, list):
            if any(h_text == ans.strip() for ans in r):
                matches += 1
        else:
            if (r or "").strip() == h_text:
                matches += 1
    return matches / total if total > 0 else 0.0
