import abc
import json
import os
import re
import unicodedata

from collections import Counter
from typing import Dict
from typing import List
from typing import Sequence
from typing import Tuple

from s2s.path import EXP_PATH


class BaseTknzr(abc.ABC):
    file_name = 'tknzr.json'
    tknzr_name = 'base'

    def __init__(self, cfg: Dict):
        self.is_cased = cfg['is_cased']
        self.min_count = cfg['min_count']
        self.n_vocab = cfg['n_vocab']
        self.bos_tk = '[bos]'
        self.eos_tk = '[eos]'
        self.pad_tk = '[pad]'
        self.unk_tk = '[unk]'
        self.tk2id = {}
        self.id2tk = {}

        for (
                sp_tk_id,
                sp_tk,
        ) in enumerate([self.pad_tk, self.bos_tk, self.eos_tk, self.unk_tk]):
            self.tk2id[sp_tk] = sp_tk_id
            self.id2tk[sp_tk_id] = sp_tk

    def preprocess(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', text)).strip()

        if not self.is_cased:
            text = text.lower()

        return text

    @abc.abstractmethod
    def tknz(self, text: str) -> List[str]:
        raise NotImplementedError

    @abc.abstractmethod
    def dtknz(self, tks: Sequence[str]) -> str:
        raise NotImplementedError

    def enc(self, text: str) -> Tuple[List[int], int]:
        r"""Encode text."""
        tk_ids = [self.tk2id[self.bos_tk]]
        for tk in self.tknz(text=self.preprocess(text=text)):
            if tk in self.tk2id:
                tk_ids.append(self.tk2id[tk])
            else:
                tk_ids.append(self.tk2id[self.unk_tk])

        tk_ids.append(self.tk2id[self.eos_tk])

        return tk_ids, len(tk_ids)

    def dec(self, tk_ids: Sequence[int]) -> str:
        r"""Decode text."""
        tks = [
            self.id2tk[tk_id]
            for tk_id in tk_ids
            if tk_id not in [
                self.tk2id[self.bos_tk],
                self.tk2id[self.eos_tk],
                self.tk2id[self.pad_tk],
            ]
        ]
        return self.dtknz(tks=tks)

    def batch_enc(
            self,
            batch_text: Sequence[str],
            max_len: int,
    ) -> Tuple[List[List[int]], List[int]]:
        r"""Batch encode text."""
        batch_tk_ids = []
        batch_tk_ids_len = []
        for text in batch_text:
            tk_ids, tk_ids_len = self.enc(text=text)

            # Pad to max sequence length.
            tk_ids = (
                tk_ids +
                [self.tk2id[self.pad_tk]] * (max_len - len(tk_ids))
            )

            # Truncate to max sequence length.
            tk_ids = tk_ids[:max_len]

            batch_tk_ids.append(tk_ids)
            batch_tk_ids_len.append(min(len(tk_ids), tk_ids_len))

        return batch_tk_ids, batch_tk_ids_len

    def batch_dec(
            self,
            batch_tk_ids: Sequence[Sequence[int]],
    ) -> List[str]:
        r"""Batch decode text."""
        return [self.dec(tk_ids=tk_ids) for tk_ids in batch_tk_ids]

    def build_vocab(self, batch_text: Sequence[str]) -> None:
        c = Counter()
        for text in batch_text:
            c.update(self.tknz(self.preprocess(text)))

        max_id = len(self.tk2id)
        for tk, tk_count in c.most_common():
            if max_id >= self.n_vocab:
                break

            if tk_count < self.min_count:
                continue

            if tk in self.tk2id:
                continue

            self.tk2id[tk] = max_id
            self.id2tk[max_id] = tk
            max_id += 1

    def save(self, exp_name: str) -> None:
        exp_path = os.path.join(EXP_PATH, exp_name)
        file_path = os.path.join(exp_path, self.__class__.file_name)

        if not os.path.exists(exp_path):
            os.makedirs(exp_path)

        with open(file_path, 'w', encoding='utf-8') as tknzr_file:
            json.dump(self.tk2id, tknzr_file, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, cfg: Dict) -> 'BaseTknzr':
        file_path = os.path.join(EXP_PATH, cfg['exp_name'], cls.file_name)

        if not os.path.exists(file_path):
            raise FileNotFoundError(f'{file_path} does not exist.')

        self = cls(cfg=cfg)
        with open(file_path, 'r', encoding='utf-8') as tknzr_file:
            self.tk2id = json.load(tknzr_file)
        self.id2tk = {tk_id: tk for tk, tk_id in self.tk2id.items()}

        return self
