from torch.utils.data import Dataset
from datasets import load_dataset
from itertools import islice

class CodeCompletionDataset(Dataset):
    def __init__(self, split="test", num_examples=5):
        actual_split = "test"
        ds_stream = load_dataset(
            "google-research-datasets/mbpp",
            "full",
            split=actual_split,
            streaming=True,
        )
        examples = list(islice(ds_stream, num_examples))

        self.items = []
        for ex in examples:
            prompt = ex.get("text", "")
            code = ex.get("code", "")
            self.items.append({"prompt": prompt, "reference": code})

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]
