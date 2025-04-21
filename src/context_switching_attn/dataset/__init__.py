from context_switching_attn.dataset.mmlu import MMLUDataset, get_mmlu_dataloader

__all__ = ["MMLUDataset", "get_mmlu_dataloader"]


dataloader = get_mmlu_dataloader("dev", "anatomy", num_concatenated=2)


for batch in dataloader:
    print(batch)
    break
