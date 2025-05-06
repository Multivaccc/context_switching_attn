from torch.utils.data import Dataset
import datasets

class ParaphraseDataset(Dataset):
    def __init__(self, split="test", num_examples=None):
        split_spec = f"{split}[:{num_examples}]" if num_examples else split
        raw = datasets.load_dataset("paws", "labeled_final", split=split_spec)
        examples = list(raw)
        filtered = [ex for ex in examples if ex["label"] == 1]
        self.items = [
            {
                "prompt": ex["sentence1"],
                "reference": ex["sentence2"],
            }
            for ex in filtered
        ]
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, i):
        return self.items[i]