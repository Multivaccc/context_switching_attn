from typing import Any, Dict, List, Tuple

from .dataset.code_completion import CodeCompletionDataset
from .dataset.gigaword import GigawordDataset
from .dataset.mmlu import MMLUDataset
from .dataset.paraphrase import ParaphraseDataset
from .dataset.rotten_tomatoes import RottenTomatoesDataset
from .dataset.tweet_qa import TweetQADataset

# map the name you pass on the CLI to the Dataset class
_DATASETS = {
    "code_completion": CodeCompletionDataset,
    "gigaword": GigawordDataset,
    "mmlu": MMLUDataset,
    "paraphrase": ParaphraseDataset,
    "rotten_tomatoes": RottenTomatoesDataset,
    "tweet_qa": TweetQADataset,
}

class PromptLoader:
    """
    Loads in-context -chat- histories and evaluation splits
    for any of your six datasets.
    """

    def __init__(self, incontext: str, eval: str):
        if incontext not in _DATASETS or eval not in _DATASETS:
            raise ValueError(f"Unknown dataset: choose from {_DATASETS.keys()}")
        self._InCtx = _DATASETS[incontext]
        self._Eval  = _DATASETS[eval]

    def load_incontext(self, num_examples: int) -> List[Dict[str, Any]]:
        """
        Return a list of {"role","content"} turns for the first
        num_examples from the TRAIN split of the incontext dataset.
        """
        if num_examples <= 0:
            return []

        ds = self._InCtx(split="train", num_examples=num_examples)
        turns: List[Dict[str, Any]] = []
        for item in ds:
            turns.append({"role": "user", "content": item["prompt"]})
            # either a generative reference-
            if "reference" in item:
                turns.append({"role": "assistant", "content": item["reference"]})
            # -or a choice-based label
            elif "choices" in item and "label" in item:
                correct = item["choices"][item["label"]]
                turns.append({"role": "assistant", "content": correct})
            else:
                raise ValueError("Dataset item missing 'reference' or 'choices'/'label'")
        return turns

    def load_eval(
        self, eval_size: int = None
    ) -> Tuple[List[str], List[Any], List[List[str]]]:
        """
        Return three lists over the TEST split:
          1) prompts:    list of the user-side prompt strings
          2) references: ground-truth (string for gen, int for choice)
          3) choices:    None for gen, or list[str] for choice
        """
        # if eval_size is None many of your datasets default to fixed test-size
        ds = (
            self._Eval(split="test", num_examples=eval_size)
            if eval_size is not None
            else self._Eval(split="test")
        )

        prompts, refs, choices = [], [], []
        for item in ds:
            prompts.append(item["prompt"])
            if "reference" in item:
                refs.append(item["reference"])
                choices.append(None)
            elif "choices" in item and "label" in item:
                refs.append(item["label"])
                choices.append(item["choices"])
            else:
                raise ValueError("Eval item missing 'reference' or 'choices'/'label'")
        return prompts, refs, choices
