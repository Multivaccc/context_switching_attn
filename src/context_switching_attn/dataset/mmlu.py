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
        split_spec = f"{split}[:{num_examples}]" if num_examples else split
        ds = load_dataset("cais/mmlu", subjects, split=split_spec)
        # now build a simple list of {prompt, choices, label}
        self.items: List[Dict] = []
        for ex in ds:
            q   = ex["question"]            # string
            c   = ex["choices"]             # list[str]
            ans = ex["answer"]              # e.g. "A" or integer index
            # Map ClassLabel string to 0-based index if needed:
            if isinstance(ans, str):
                lbl = ord(ans) - ord("A")
            else:
                lbl = int(ans)
            self.items.append({
                "prompt":  q,
                "choices": c,
                "label":   lbl,
            })

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> Dict:
        return self.items[i]
