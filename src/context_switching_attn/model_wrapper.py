import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class ModelWrapper:
    """
    Wraps a causal LM for both classification (choice-based) and generative tasks.
    For classification, we score each choice by log-likelihood and pick the highest.
    """

    def __init__(self, model_name: str, device: str | None = None, blackbox: bool = False):
        self.blackbox = blackbox

        if self.blackbox:
            from openrouter import OpenRouterClient
            self.client = OpenRouterClient(model_name)
            return
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
        )
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.to(self.device).eval()

    def _build_prompt(self, history: list[dict]) -> str:
        # history: list of {"role": "user"/"assistant", "content": ...}
        s = ""
        for turn in history:
            role = "User" if turn["role"] == "user" else "Assistant"
            s += f"{role}: {turn['content']}\n"
        return s

    def classify(self, history: list[dict], choices: list[str]):
        if self.blackbox:
            return self.client.classify(history, choices)
        """
        Returns: (pred_index, confidence, all_probs: list[float])
        """
        prompt = self._build_prompt(history)
        device = self.device

        # accumulate log-likelihood for each choice
        llhs = []
        for choice in choices:
            text = prompt + choice
            enc = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=1024,
            ).to(device)
            with torch.no_grad():
                outputs = self.model(**enc, labels=enc.input_ids)
                # negative loss is sum log-probs over all tokens
                llh = -outputs.loss.item() * enc.input_ids.shape[-1]
            llhs.append(llh)

        probs = torch.softmax(torch.tensor(llhs), dim=0)
        pred = int(probs.argmax().item())
        return pred, float(probs[pred].item()), [float(p) for p in probs]

    def generate(self, history: list[dict], max_new_tokens: int = 64, pad_token_id: int = None):
        if self.blackbox:
            return self.client.generate(history)
        """
        Returns the raw generated string (for generative tasks).
        """
        prompt = self._build_prompt(history)
        enc = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        if pad_token_id is None:
            pad_token_id = self.tokenizer.pad_token_id
        with torch.no_grad():
            out = self.model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                pad_token_id=pad_token_id  # suppress warning
            )
        gen = self.tokenizer.decode(out[0, enc.input_ids.shape[-1]:], skip_special_tokens=True)
        return gen.strip()
