r"""Configuration for sequence to sequence model."""

import abc
import argparse
import json
import os

from typing import List

from s2s.dset import DSET_OPTS
from s2s.path import EXP_PATH


class CfgMixin(abc.ABC):
    file_name = 'cfg.json'

    @abc.abstractmethod
    def __init__(self, **kwargs):
        raise NotImplementedError

    def save(self, exp_name: str) -> None:
        exp_path = os.path.join(EXP_PATH, exp_name)
        file_path = os.path.join(exp_path, self.__class__.file_name)

        if not os.path.exists(exp_path):
            os.makedirs(exp_path)

        with open(file_path, 'w', encoding='utf-8') as cfg_file:
            json.dump(
                self.__dict__,
                cfg_file,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )

    @classmethod
    def load(cls, exp_name: str) -> 'CfgMixin':
        file_path = os.path.join(EXP_PATH, exp_name, cls.file_name)

        if not os.path.exists(file_path):
            raise FileNotFoundError(f'{file_path} does not exist.')

        with open(file_path, 'r', encoding='utf-8') as cfg_file:
            return cls(**json.load(cfg_file))

    @classmethod
    @abc.abstractmethod
    def update_parser(cls, parser: argparse.ArgumentParser) -> None:
        raise NotImplementedError


class BaseExpCfg(CfgMixin):
    file_name = 'exp_cfg.json'

    def __init__(self, **kwargs):
        self.exp_name = kwargs['exp_name']

    @classmethod
    def update_parser(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            '--exp_name',
            help='Current experiment name.',
            required=True,
            type=str,
        )


class BaseModelCfg(BaseExpCfg):
    file_name = 'model_cfg.json'
    model_name = 'Base'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dec_tknzr_exp = kwargs['dec_tknzr_exp']
        self.enc_tknzr_exp = kwargs['enc_tknzr_exp']

    @classmethod
    def update_parser(cls, parser: argparse.ArgumentParser) -> None:
        super().update_parser(parser=parser)

        parser.add_argument(
            '--dec_tknzr_exp',
            help='Experiment name of the decoder paired tokenizer.',
            required=True,
            type=str,
        )
        parser.add_argument(
            '--enc_tknzr_exp',
            help='Experiment name of the encoder paired tokenizer.',
            required=True,
            type=str,
        )


class BaseOptimCfg(CfgMixin):
    file_name = 'optim_cfg.json'

    def __init__(self, **kwargs):
        self.lr = kwargs['lr']

    @classmethod
    def update_parser(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            '--lr',
            help='',
            required=True,
            type=float,
        )


class BaseTknzrCfg(BaseExpCfg):
    file_name = 'tknzr_cfg.json'
    tknzr_name = 'Base'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dset_name: List = kwargs['dset_name']
        self.min_count: int = kwargs['min_count']
        self.n_vocab: int = kwargs['n_vocab']
        self.is_cased: bool = kwargs['is_cased']
        self.bos_tk: str = kwargs['bos_tk']
        self.eos_tk: str = kwargs['eos_tk']
        self.pad_tk: str = kwargs['pad_tk']
        self.unk_tk: str = kwargs['unk_tk']
        self.tknzr_name: str = self.__class__.tknzr_name

    @classmethod
    def update_parser(cls, parser: argparse.ArgumentParser) -> None:
        super().update_parser(parser=parser)

        dset_choices = []
        for dset_name in DSET_OPTS:
            dset_choices.extend([
                f'{dset_name}.src',
                f'{dset_name}.tgt',
            ])

        parser.add_argument(
            '--dset_name',
            choices=dset_choices,
            help='Name(s) of the dataset(s) to perform tokenizer training experiment.',
            nargs='+',
            required=True,
        )
        parser.add_argument(
            '--min_count',
            help='',
            required=True,
            type=int,
        )
        parser.add_argument(
            '--n_vocab',
            help='Tokenizer vocabulary size.',
            required=True,
            type=int,
        )
        parser.add_argument(
            '--is_cased',
            action='store_true',
            help='',
        )
        parser.add_argument(
            '--bos_tk',
            help='',
            default='[bos]',
            type=str,
        )
        parser.add_argument(
            '--eos_tk',
            help='',
            default='[eos]',
            type=str,
        )
        parser.add_argument(
            '--pad_tk',
            help='',
            default='[pad]',
            type=str,
        )
        parser.add_argument(
            '--unk_tk',
            help='',
            default='[unk]',
            type=str,
        )

    @classmethod
    def cont_parser(cls, parser: argparse.ArgumentParser) -> None:
        r"""Continue training tokenizer."""
        super().update_parser(parser=parser)

        dset_choices = []
        for dset_name in DSET_OPTS:
            dset_choices.extend([
                f'{dset_name}.src',
                f'{dset_name}.tgt',
            ])

        parser.add_argument(
            '--dset_name',
            choices=dset_choices,
            help='Additional name(s) of the dataset(s) to continue tokenizer training experiment.',
            nargs='+',
            required=True,
        )
        parser.add_argument(
            '--n_vocab',
            help='Increase tokenizer vocabulary size.',
            required=True,
            type=int,
        )
