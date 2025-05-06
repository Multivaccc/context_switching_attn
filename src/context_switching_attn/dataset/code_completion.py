from torch.utils.data import Dataset
import datasets

class CodeCompletionDataset(Dataset):
    def __init__(self, split="test", num_examples=None):
        split_spec = f"{split}[:{num_examples}]" if num_examples else split
        ds = datasets.load_dataset("mbpp", split=split_spec)
        self.items = []
        for ex in ds:
            prompt = ex.get("text", ex.get("prompt", ""))
            code = ex.get("code", ex.get("snippet", ""))
            self.items.append({"prompt": prompt, "reference": code})
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, i):
        return self.items[i]