import abc
import json
import os
import re
import unicodedata

from collections import Counter
from typing import List
from typing import Sequence

import s2s
import s2s.path

from s2s.cfg import BaseExpCfg
from s2s.cfg import BaseTkerCfg


class BaseTker(abc.ABC):
    file_name = 'tker.json'

    def __init__(self, tker_cfg: BaseTkerCfg):
        self.is_cased = tker_cfg.is_cased
        self.min_count = tker_cfg.min_count
        self.n_vocab = tker_cfg.n_vocab
        self.bos_tk = self.preprocess(tker_cfg.bos_tk)
        self.eos_tk = self.preprocess(tker_cfg.eos_tk)
        self.pad_tk = self.preprocess(tker_cfg.pad_tk)
        self.unk_tk = self.preprocess(tker_cfg.unk_tk)
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
    def detokenize(self, tks: Sequence[str]) -> str:
        raise NotImplementedError

    def build_vocab(self, batch_src: Sequence[str]):
        c = Counter()
        for src in batch_src:
            c.update(self.tokenize(self.preprocess(src)))

        max_id = len(self.tk2id)
        for tk, tk_count in c.most_common():
            if tk_count < self.min_count:
                continue

            if tk in self.tk2id:
                continue

            self.tk2id[tk] = max_id
            self.id2tk[max_id] = tk
            max_id += 1

    def save(self, exp_cfg: BaseExpCfg):
        exp_path = os.path.join(s2s.path.EXP_PATH, exp_cfg.exp_name)
        file_path = os.path.join(exp_path, self.__class__.file_name)

        if not os.path.exists(exp_path):
            os.makedirs(exp_path)

        with open(file_path, 'w', encoding='utf-8') as tker_file:
            json.dump(self.tk2id, tker_file, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, exp_cfg: BaseExpCfg):
        file_path = os.path.join(
            s2s.path.EXP_PATH,
            exp_cfg.exp_name,
            cls.file_name,
        )

        if not os.path.exists(file_path):
            raise FileNotFoundError(f'{file_path} does not exist.')

        self = cls(cfg=cfg)
        with open(file_path, 'r', encoding='utf-8') as tker_file:
            self.tk2id = json.load(tker_file)
        self.id2tk = {tk_id: tk for tk, tk_id in self.tk2id.items()}

        return self
