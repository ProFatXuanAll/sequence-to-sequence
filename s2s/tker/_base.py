import abc
import re
import unicodedata

from typing import List
from typing import Sequence

class BaseTker(abc.ABC):

    def __init__(
            self,
            is_cased: bool,
            min_count: int,
            n_vocab: int,
            bos_tk: str='[bos]',
            eos_tk: str='[eos]',
            pad_tk: str='[pad]',
            unk_tk: str='[unk]',
    ):
        self.is_cased = is_cased
        self.min_count = min_count
        self.n_vocab = n_vocab
        self.bos_tk = self.preprocess(bos_tk)
        self.eos_tk = self.preprocess(eos_tk)
        self.pad_tk = self.preprocess(pad_tk)
        self.unk_tk = self.preprocess(unk_tk)
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
