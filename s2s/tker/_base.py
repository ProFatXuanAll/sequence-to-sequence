import abc
import re
import unicodedata

from typing import List
from typing import Sequence

from s2s.cfg import BaseTkerCfg


class BaseTker(abc.ABC):
    def __init__(self, cfg: BaseTker):
        self.is_cased = cfg.is_cased
        self.min_count = cfg.min_count
        self.n_vocab = cfg.n_vocab
        self.bos_tk = self.preprocess(cfg.bos_tk)
        self.eos_tk = self.preprocess(cfg.eos_tk)
        self.pad_tk = self.preprocess(cfg.pad_tk)
        self.unk_tk = self.preprocess(cfg.unk_tk)
        self.tk2id = {}
        self.id2tk = {}

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
        for src in batch_src:
            for tk in self.tokenize(self.preprocess(src)):
                pass
