import re
import unicodedata
import torch


class BaseDset(torch.utils.data.Dataset):
    def __init__(self):
        self.src = []
        self.tgt = []

    def all_src(self):
        return self.src

    def all_tgt(self):
        return self.tgt

    @classmethod
    def preprocess(cls, text: str) -> str:
        return re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', text)).strip()
