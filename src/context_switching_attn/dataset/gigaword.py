from torch.utils.data import Dataset
from datasets import load_dataset
from itertools import islice

class GigawordDataset(Dataset):
    def __init__(self, split="test", num_examples=5):
        ds_stream = load_dataset(
            "gigaword",
            split=split,
            streaming=True,
            trust_remote_code=True,
        )
        examples = list(islice(ds_stream, num_examples))

        self.items = [
            {"prompt": ex["document"], "reference": ex["summary"]}
            for ex in examples
        ]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]
