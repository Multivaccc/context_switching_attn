from torch.utils.data import Dataset
from datasets import load_dataset

class GigawordDataset(Dataset):
    def __init__(self, split="test", num_examples=None):
        # only load the first num_examples documents
        split_spec = f"{split}[:{num_examples}]" if num_examples else split
        ds = load_dataset("gigaword", split=split_spec, trust_remote_code=True)
        self.items = [{"prompt": ex["document"], "reference": ex["summary"]} for ex in ds]
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, i):
        return self.items[i]