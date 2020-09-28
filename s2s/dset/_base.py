import re
import unicodedata
import torch


class BaseDset(torch.utils.data.Dataset):
    def __init__(self, is_cased: bool):
        self.is_cased = is_cased

    def preprocess(self, text: str):
        text = re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', text)).strip()

        if not self.is_cased:
            text = text.lower()

        return text
