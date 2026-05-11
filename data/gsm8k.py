import re

from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset


def _extract_reference(answer_text: str) -> str | None:
    m = re.search(r"####\s*([\-\d,\.]+)", answer_text)
    return m.group(1).replace(",", "") if m else None


class GSM8KDataset(Dataset):
    def __init__(self, tokenizer, split: str = "train"):
        raw = load_dataset("gsm8k", "main", split=split)
        self.tokenizer = tokenizer
        self.data = [
            {
                "prompt": self._make_prompt(item["question"]),
                "reference": _extract_reference(item["answer"]),
            }
            for item in raw
            if _extract_reference(item["answer"]) is not None
        ]

    def _make_prompt(self, question: str) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a math assistant. Solve the problem step by step. "
                    "Write your final numerical answer after '####'."
                ),
            },
            {"role": "user", "content": question},
        ]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        return self.data[idx]


def make_dataloader(dataset: GSM8KDataset, batch_size: int) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
