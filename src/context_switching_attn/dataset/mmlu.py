from typing import Dict, List, Optional
from torch.utils.data import Dataset
import datasets

class MMLUDataset(Dataset):
    def __init__(
        self,
        split: str = "test",
        subjects: str = "all",
        num_examples: Optional[int] = None,
    ):
        split_spec = f"{split}[:{num_examples}]" if num_examples else split
        ds = datasets.load_dataset(path="lukaemon/mmlu", name=subjects, split=split_spec)

        # now build a simple list of {prompt, choices, label}
        self.items: List[Dict] = []
        for ex in ds:
            q = ex["input"]            # string
            c = [ex["A"], ex["B"], ex["C"], ex["D"]]  # list[str]
            ans = ex["target"]         # e.g. "A", "B", "C", or "D"
            lbl = ord(ans) - ord("A")
            self.items.append({
                "prompt":  q,
                "choices": c,
                "label":   lbl,
            })

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> Dict:
        return self.items[i]
