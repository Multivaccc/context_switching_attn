from typing import Dict, List, Optional
from torch.utils.data import Dataset
from datasets import load_dataset

class MMLUDataset(Dataset):
    """
    Dataset class for MMLU (Massive Multitask Language Understanding),
    in the same style as your other datasets:

      - __init__(split, subjects, num_examples)
      - truncate via select(range(num_examples))
      - self.items = list of dicts with keys prompt, choices, label
    """

    def __init__(
        self,
        split: str = "test",
        subjects: str = "all",
        num_examples: Optional[int] = None,
    ):
        """
        Args:
          split: 'train' / 'validation' / 'test'
          subjects: e.g. 'abstract_algebra' or 'all'
          num_examples: cap how many examples to load
        """
        # load the raw HF dataset (split=name, config=subjects)
        ds = load_dataset("cais/mmlu", subjects, split=split)
        if num_examples:
            ds = ds.select(range(num_examples))

        # now build a simple list of {prompt, choices, label}
        self.items: List[Dict] = []
        for ex in ds:
            question = ex["question"]            # string
            choices  = ex["choices"]             # list[str]
            label    = int(ex["answer"])         # int
            self.items.append({
                "prompt": question,
                "choices": choices,
                "label":   label,
            })

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> Dict:
        return self.items[i]
