import os

from typing import Sequence

import nltk
import pandas as pd
from sklearn.metrics import accuracy_score

from s2s.dset._base import BaseDset
from s2s.path import DATA_PATH


class NewsTranslateDset(BaseDset):
    dset_name = 'news_translate'

    def __init__(self):
        super().__init__()
        with open(
            os.path.join(DATA_PATH, 'news-commentary-v12.zh-en.en'),
            'r',
            encoding='utf8'
        ) as input_src_file:
            src = input_src_file.readlines()
        with open(
            os.path.join(DATA_PATH, 'news-commentary-v12.zh-en.zh'),
            'r',
            encoding='utf8'
        ) as input_tgt_file:
            tgt = input_tgt_file.readlines()

        for i in range(len(src)):
            self.src.append(self.__class__.preprocess(src[i]))
            self.tgt.append(self.__class__.preprocess(tgt[i]))

    @staticmethod
    def batch_eval(
            batch_tgt: Sequence[str],
            batch_pred: Sequence[str],
    ) -> float:
        batch_tgt = [[k for k in i] for i in batch_tgt]
        batch_pred = [[[k for k in i]] for i in batch_pred]
        return nltk.translate.bleu_score.corpus_bleu(
            batch_pred,
            batch_tgt
        )
