r"""Configuration for sequence to sequence model."""

import abc
import argparse
import json
import os

import s2s.path


class CfgMixin(abc.ABC):
    file_name = 'cfg.json'

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            self.__dict__[key] = value

    def save(self, exp_name: str) -> None:
        exp_path = os.path.join(s2s.path.EXP_PATH, exp_name)
        file_path = os.path.join(exp_path, self.__class__.file_name)

        if not os.path.exists(exp_path):
            os.makedirs(exp_path)

        with open(file_path, 'w', encoding='utf-8') as cfg_file:
            json.dump(self.__dict__, cfg_file, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, exp_name: str) -> 'CfgMixin':
        file_path = os.path.join(s2s.path.EXP_PATH, exp_name, cls.file_name)

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

    @classmethod
    def update_parser(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            '--exp_name',
            help='Current experiment name.',
            required=True,
            type=str,
        )


class BaseModelCfg(CfgMixin):
    file_name = 'model_cfg.json'
    model_name = 'Base'


class BaseOptimCfg(CfgMixin):
    file_name = 'optim_cfg.json'

    @classmethod
    def update_parser(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            '--lr',
            help='',
            required=True,
            type=float,
        )


class BaseTkerCfg(CfgMixin):
    file_name = 'tker_cfg.json'

    @classmethod
    def update_parser(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            '--min_count',
            help='',
            required=True,
            type=int,
        )
        parser.add_argument(
            '--n_vocab',
            help='',
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
