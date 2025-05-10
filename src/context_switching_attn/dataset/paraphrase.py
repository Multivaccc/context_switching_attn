from torch.utils.data import Dataset
from datasets import load_dataset

class ParaphraseDataset(Dataset):
    def __init__(self, split="test", num_examples=5):
        # always limit to exactly num_examples
        ds_stream = load_dataset(
            "google-research-datasets/paws",
            "labeled_final",
            split=split,
            streaming=True,
        )
        # only keep the first num_examples _positive_ pairs
        filtered = []
        for ex in ds_stream:
            if ex["label"] == 1:
                filtered.append(ex)
            if len(filtered) >= num_examples:
                break

        self.items = [
            {"prompt": ex["sentence1"], "reference": ex["sentence2"]}
            for ex in filtered
        ]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]
