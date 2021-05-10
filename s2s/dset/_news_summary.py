import os
from io import TextIOWrapper
from typing import Sequence
from zipfile import ZipFile

import nltk
import pandas as pd
import requests

from s2s.dset._base import BaseDset
from s2s.path import DATA_PATH


class NewsSummaryDset(BaseDset):
    dset_name = 'news_summary'
    url = ''.join([
        'https://github.com/ProFatXuanAll',
        '/demo-dataset/raw/main/news_summary',
    ])

    def __init__(self):
        super().__init__()
        filename = f'news_summary.csv'
        file_path = os.path.join(
            DATA_PATH,
            filename + '.zip'
        )
        url = f'{self.__class__.url}/{filename}.zip'

        # Check if file exist. If file not exist then download the file.
        self.download(url, file_path)

        with ZipFile(file_path, 'r') as input_zipfile:
            with TextIOWrapper(
                input_zipfile.open(filename, 'r'),
                encoding='iso-8859-1'
            ) as input_text_file:
                df = pd.read_csv(
                    input_text_file
                ).dropna()

        src = df['ctext'].apply(self.convert_to_utf8)
        tgt = df['text'].apply(self.convert_to_utf8)
        self.src.extend(
            src.apply(str).apply(self.__class__.preprocess).to_list()
        )
        self.tgt.extend(
            tgt.apply(str).apply(self.__class__.preprocess).to_list()
        )

    def download(self, url: str, file_path: str) -> None:
        file_dir = os.path.abspath(os.path.join(file_path, os.pardir))

        if os.path.exists(file_path):
            return
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)

        with requests.get(url) as res, open(file_path, 'wb') as out_file:
            out_file.write(res.content)

    def convert_to_utf8(self, sample: str):
        return sample.encode('iso-8859-1').decode('utf8', 'ignore')

    @staticmethod
    def batch_eval(
            batch_tgt: Sequence[str],
            batch_pred: Sequence[str],
    ) -> float:
        batch_tgt = [[[k for k in i]] for i in batch_tgt]
        batch_pred = [[k for k in i] for i in batch_pred]

        return nltk.translate.bleu_score.corpus_bleu(
            batch_tgt,
            batch_pred
        )
