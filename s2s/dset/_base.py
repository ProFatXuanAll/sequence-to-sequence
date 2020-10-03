import re
import unicodedata
import torch

from typing import Tuple

class BaseDset(torch.utils.data.Dataset):
    def __init__(self):
        self.src = []
        self.tgt = []

    def __len__(self) -> int:
        return len(self.tgt)

    def __getitem__(self, index: int) -> Tuple[str, str]:
        return self.src[index], self.tgt[index]

    def all_src(self):
        return self.src

    def all_tgt(self):
        return self.tgt

    @classmethod
    def preprocess(cls, text: str) -> str:
        return re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', text)).strip()
