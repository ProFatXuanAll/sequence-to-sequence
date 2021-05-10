import os
from io import TextIOWrapper
from typing import Sequence
from zipfile import ZipFile

import nltk
import requests

from s2s.dset._base import BaseDset
from s2s.path import DATA_PATH


class BaseNewsTranslateDset(BaseDset):
    dset_name = 'base_news_translate'
    url = ''.join([
        'https://github.com/ProFatXuanAll',
        '/demo-dataset/raw/main/news_commentary-v13',
    ])

    def __init__(self, src, tgt):
        super().__init__()
        language = src if tgt == 'en' else tgt
        src_filename = f'news-commentary-v13.{language}-en.{src}'
        tgt_filename = f'news-commentary-v13.{language}-en.{tgt}'
        src_file_path = os.path.join(
            DATA_PATH,
            'WMT19_dset',
            src_filename + '.zip'
        )
        tgt_file_path = os.path.join(
            DATA_PATH,
            'WMT19_dset',
            tgt_filename + '.zip'
        )
        src_url = f'{self.__class__.url}/{src_filename}.zip'
        tgt_url = f'{self.__class__.url}/{tgt_filename}.zip'

        # Check if src file exist. If file not exist then download src file.
        self.download(src_url, src_file_path)
        # Check if tgt file exist. If file not exist then download tgt file.
        self.download(tgt_url, tgt_file_path)

        with ZipFile(src_file_path, 'r') as input_zipfile:
            with TextIOWrapper(
                input_zipfile.open(src_filename, 'r'),
                encoding='utf-8'
            ) as input_src_file:
                src = input_src_file.readlines()
        with ZipFile(tgt_file_path, 'r') as input_zipfile:
            with TextIOWrapper(
                input_zipfile.open(tgt_filename, 'r'),
                encoding='utf-8'
            ) as input_tgt_file:
                tgt = input_tgt_file.readlines()

        for i in range(len(src)):
            self.src.append(self.__class__.preprocess(src[i]))
            self.tgt.append(self.__class__.preprocess(tgt[i]))

    def download(self, url: str, file_path: str) -> None:
        file_dir = os.path.abspath(os.path.join(file_path, os.pardir))

        if os.path.exists(file_path):
            return
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)

        with requests.get(url) as res, open(file_path, 'wb') as out_file:
            out_file.write(res.content)

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
