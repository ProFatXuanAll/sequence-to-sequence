import os

from typing import Sequence

import nltk
import pandas as pd
from sklearn.metrics import accuracy_score

from s2s.dset._base import BaseDset
from s2s.path import DATA_PATH


class NewsSummaryDset(BaseDset):
    dset_name = 'news_summary'

    def __init__(self):
        super().__init__()
        df = pd.read_csv(
            os.path.join(DATA_PATH, 'news_summary.csv'),
            encoding='iso-8859-1'
        ).dropna()

        src = df['ctext'].apply(self.convert_to_utf8)
        tgt = df['text'].apply(self.convert_to_utf8)
        self.src.extend(
            src.apply(str).apply(self.__class__.preprocess).to_list()
        )
        self.tgt.extend(
            tgt.apply(str).apply(self.__class__.preprocess).to_list()
        )

    def convert_to_utf8(self, sample: str):
        return sample.encode('iso-8859-1').decode('utf8', 'ignore')

    @staticmethod
    def eval(tgt: str, pred: str) -> float:
        return return nltk.translate.bleu_score.sentence_bleu(
            [pred],
            tgt
        )


    @staticmethod
    def batch_eval(
            batch_tgt: Sequence[str],
            batch_pred: Sequence[str],
    ) -> float:
        return nltk.translate.bleu_score.corpus_bleu(
            batch_pred,
            batch_tgt
        )

