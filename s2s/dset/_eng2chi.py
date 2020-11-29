import os

from typing import Sequence

import pandas as pd
from sklearn.metrics import accuracy_score
from nltk.translate.bleu_score import sentence_bleu

from s2s.dset._base import BaseDset
from s2s.path import DATA_PATH

class Eng2ChiDset(BaseDset):
    r"""A class to deal with English to chinese tranlation corpus

    corpus source:http://www.manythings.org/anki/?fbclid=IwAR1EQ4HB-8jA6A5zD-167ghXqJlAiTQvzFBMJliRFQHdi0kSEeZvpO9vAK8

    Usages:
        In _init_.py : Add class into dictionary.
        
        from s2s.dset._eng2chi import Eng2ChiDset
        Dset = Union[
            ArithDset,
            Eng2ChiDset,
        ]

        DSET_OPTS: Dict[str, Type[Dset]] = {
            ArithDset.dset_name: ArithDset,
            Eng2ChiDset.dset_name: Eng2ChiDset
        }

        ----------

        In other class : Connect the args.dset_name with correspond object and use specific function

        from s2s.dset import DSET_OPTS
        dset = DSET_OPTS[args.dset_name]()

        print(DSET_OPTS[args.dset_name].batch_eval(
        batch_tgt=dset.all_tgt(),
        batch_pred=all_pred,
    ))
    """
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
        r"""
            Counting BLEU score for this translation task, instead of exact match.
        """
        return sentence_bleu(
            batch_pred,
            batch_tgt,
            smoothing_function=nltkbleu.SmoothingFunction(epsilon=1e-12).method1,
        ) 
