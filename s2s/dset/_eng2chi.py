import os

from typing import Sequence

import pandas as pd
from sklearn.metrics import accuracy_score
from nltk.translate.bleu_score import sentence_bleu

from s2s.dset._base import BaseDset
from s2s.path import DATA_PATH

try:
    from nltk.translate import bleu_score as nltkbleu
except ImportError:
    # User doesn't have nltk installed, so we can't use it for bleu
    # We'll just turn off things, but we might want to warn the user
    nltkbleu = None

class Eng2ChiDset(BaseDset):
    dset_name = 'eng2chi'

    def __init__(self):
        super().__init__()
        df = pd.read_csv(os.path.join(DATA_PATH, 'eng2chi.csv'))
        self.src.extend(
            df['src'].apply(str).apply(self.__class__.preprocess).to_list()
        )
        self.tgt.extend(
            df['tgt'].apply(str).apply(self.__class__.preprocess).to_list()
        )

    @staticmethod
    def eval(tgt: str, pred: str) -> float:
        return float(tgt == pred)

    @staticmethod
    def batch_eval(
            batch_tgt: Sequence[str],
            batch_pred: Sequence[str],
    ) -> float:
        return accuracy_score(batch_tgt, batch_pred)

    @staticmethod
    def bleu_score(
            batch_tgt: Sequence[str],
            batch_pred: Sequence[str],
    ) -> float:
        return sentence_bleu(
            batch_pred,
            batch_tgt,
            smoothing_function=nltkbleu.SmoothingFunction(epsilon=1e-12).method1,
        ) 
