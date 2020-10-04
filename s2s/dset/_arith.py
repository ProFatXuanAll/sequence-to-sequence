import os

from typing import Sequence

import pandas as pd
from sklearn.metrics import accuracy_score

from s2s.dset._base import BaseDset
from s2s.path import DATA_PATH


class ArithDset(BaseDset):
    dset_name = 'arithmetic'

    def __init__(self):
        super().__init__()
        df = pd.read_csv(os.path.join(DATA_PATH, 'arithmetic.csv'))
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
