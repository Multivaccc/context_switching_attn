from torch.utils.data import Dataset
from datasets import load_dataset

class TweetQADataset(Dataset):
    def __init__(self, split="test", num_examples=None):
        # explicit repo to get the right schema
        ds = load_dataset("ucsbnlp/tweet_qa", split=split)
        if num_examples:
            ds = ds.select(range(num_examples))

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
