from torch.utils.data import Dataset
from datasets import load_dataset

class TweetQADataset(Dataset):
    def __init__(self, split="test", num_examples=None):
        split_spec = f"{split}[:{num_examples}]" if num_examples else split
        ds = load_dataset("ucsbnlp/tweet_qa", split=split_spec)
        self.items = []
        for ex in ds:
            tweet    = ex.get("Tweet", "")
            question = ex.get("Question", "")
            answers  = ex.get("Answer", [])
            # take the first answer if available
            reference = answers[0] if isinstance(answers, list) and answers else ""
            prompt = f"{tweet} Question: {question}"
            self.items.append({"prompt": prompt, "reference": reference})

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]
