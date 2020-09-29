import abc
import re
import unicodedata

from collections import Counter
from typing import List
from typing import Sequence

from s2s.cfg import BaseTkerCfg


class BaseTker(abc.ABC):
    def __init__(self, cfg: BaseTkerCfg):
        self.is_cased = cfg.is_cased
        self.min_count = cfg.min_count
        self.n_vocab = cfg.n_vocab
        self.bos_tk = self.preprocess(cfg.bos_tk)
        self.eos_tk = self.preprocess(cfg.eos_tk)
        self.pad_tk = self.preprocess(cfg.pad_tk)
        self.unk_tk = self.preprocess(cfg.unk_tk)
        self.tk2id = {}
        self.id2tk = {}

        self.build_vocab([self.bos_tk, self.eos_tk, self.pad_tk, self.unk_tk])

    def preprocess(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', text)).strip()

        if not self.is_cased:
            text = text.lower()

        return text

    @abc.abstractmethod
    def tokenize(self, text: str) -> List[str]:
        raise NotImplementedError

    @abc.abstractmethod
    def detokenize(self, tks: List[str]) -> str:
        raise NotImplementedError

    def build_vocab(self, batch_src: Sequence[str]):
        counter = Counter()
        for src in batch_src:
            counter.update(self.tokenize(self.preprocess(src)))

        max_id = len(self.tk2id)
        for token, token_count in counter.most_common():
            if token_count < self.min_count:
                continue

            if token in self.tk2id:
                continue

            self.tk2id[token] = max_id
            self.id2tk[max_id] = token
            max_id += 1
