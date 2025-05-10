from typing import Dict, List
from torch.utils.data import Dataset
import datasets

class MMLUDataset(Dataset):
    def __init__(
        self,
        split: str = "test",
        subjects: str = "abstract_algebra",
        num_examples: int = 5,
    ):
        # Map common aliases to HF split names
        alias_map = {"dev": "validation", "val": "validation"}
        hf_split = alias_map.get(split, split)
        if hf_split not in ("train", "validation", "test"):
            raise ValueError(
                f'Unknown MMLU split "{split}". Choose from train, validation, or test.'
            )

        # Try a sliced load; if slicing is out of range, fall back to full+select
        split_spec = f"{hf_split}[:{num_examples}]"
        try:
            ds = datasets.load_dataset(
                "lukaemon/mmlu",
                name=subjects,
                split=split_spec,
                trust_remote_code=True,
            )
        except ValueError:
            full = datasets.load_dataset(
                "lukaemon/mmlu",
                name=subjects,
                split=hf_split,
                trust_remote_code=True,
            )
            count = min(len(full), num_examples)
            ds = full.select(range(count))

        # Build items list
        self.items: List[Dict] = []
        for ex in ds:
            q = ex["input"]
            c = [ex["A"], ex["B"], ex["C"], ex["D"]]
            ans = ex["target"]  # "A", "B", "C", or "D"
            lbl = ord(ans) - ord("A")
            self.items.append({
                "prompt": q,
                "choices": c,
                "label": lbl,
            })

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> Dict:
        return self.items[i]
