import os
import requests
import json
from typing import List, Dict, Tuple


class OpenRouterClient:
    def __init__(self, model_name: str):
        self.model_name = model_name
        # self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.api_key = "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"

        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def _build_prompt(self, history: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Convert internal message history to OpenRouter-compatible format.
        """
        return [{"role": turn["role"], "content": turn["content"]} for turn in history]

    def generate(self, history: List[Dict[str, str]], max_tokens: int = 64) -> str:
        messages = self._build_prompt(history)

        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.0
        }

        response = requests.post(self.api_url, headers=self.headers, json=payload)
        if response.status_code != 200:
            raise RuntimeError(f"OpenRouter API error {response.status_code}: {response.text}")

        data = response.json()
        return data["choices"][0]["message"]["content"].strip()

    def classify(self, history: List[Dict[str, str]], choices: List[str]) -> Tuple[int, float, List[float]]:
        """
        Simulate classification by appending each choice and scoring via log-likelihood approximation.
        For now, use generation and match against choices (very approximate).
        """
        prompt = self._build_prompt(history)
        prompt.append({
            "role": "user",
            "content": "Choose one of the following:\n" + "\n".join(f"{i}. {c}" for i, c in enumerate(choices))
        })

        payload = {
            "model": self.model_name,
            "messages": prompt,
            "max_tokens": 16,
            "temperature": 0.0
        }

        response = requests.post(self.api_url, headers=self.headers, json=payload)
        if response.status_code != 200:
            raise RuntimeError(f"OpenRouter API error {response.status_code}: {response.text}")

        content = response.json()["choices"][0]["message"]["content"].strip().lower()

        # Try to find best match
        best = 0
        for i, choice in enumerate(choices):
            if choice.lower() in content:
                best = i
                break

        probs = [0.0] * len(choices)
        probs[best] = 1.0
        return best, 1.0, probs
