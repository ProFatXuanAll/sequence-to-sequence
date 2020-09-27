r"""Configuration for sequence to sequence model."""

import abc
import argparse
import json
import os

from typing import Union

import s2s.path

class CfgMixin(abc.ABC):
    def __iter__(self):
        for key, value in self.__dict__.items():
            yield key, value

    def save(self, exp_name: str, file_name: str) -> None:
        cfg = {
            k: v for k, v in iter(self)
            if isinstance(v, (bool, float, int, str))
        }
        exp_path = os.path.join(s2s.path.EXP_PATH, exp_name)
        file_path = os.path.join(exp_path, file_name)

        if not os.path.exists(exp_path):
            os.makedirs(exp_path)

        with open(file_path, 'w', encoding='utf-8') as cfg_file:
            json.dump(cfg, cfg_file, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, exp_name: str, file_name: str, **kwargs) -> 'CfgMixin':
        file_path = os.path.join(s2s.path.EXP_PATH, exp_name, file_name)

        if not os.path.exists(file_path):
            raise FileNotFoundError(f'{file_path} does not exist.')

        with open(file_path, 'r', encoding='utf-8') as cfg_file:
            cfg = json.load(cfg_file)

        return cls(**cfg, **kwargs)

    @classmethod
    @abc.abstractmethod
    def parse_args(cls, args: argparse.Namespace) -> 'CfgMixin':
        raise NotImplementedError

    @classmethod
    def peek_cfg_value(cls, exp_name: str, file_name: str, key: str) -> Union[
        bool,
        float,
        int,
        str,
    ]:
        file_path = os.path.join(s2s.path.EXP_PATH, exp_name, file_name)

        if not os.path.exists(file_path):
            raise FileNotFoundError(f'{file_path} does not exist.')

        with open(file_path, 'r', encoding='utf-8') as cfg_file:
            cfg = json.load(cfg_file)

        return cfg[key]

class BaseEncCfg(CfgMixin):
    def save(self, exp_name: str) -> None:
        super().save(exp_name=exp_name, file_name='enc_cfg.json')

    @classmethod
    def load(cls, exp_name: str) -> 'BaseEncCfg':
        return super().load(exp_name=exp_name, file_name='enc_cfg.json')

    @classmethod
    def peek_cfg_value(cls, exp_name: str, key: str) -> Union[
        bool,
        float,
        int,
        str,
    ]:
        return super().peek_cfg_value(
            exp_name=exp_name,
            file_name='enc_cfg.json',
            key=key
        )

class BaseDecCfg(CfgMixin):
    def save(self, exp_name: str) -> None:
        super().save(exp_name=exp_name, file_name='dec_cfg.json')

    @classmethod
    def load(cls, exp_name: str) -> 'BaseDecCfg':
        return super().load(exp_name=exp_name, file_name='dec_cfg.json')

    @classmethod
    def peek_cfg_value(cls, exp_name: str, key: str) -> Union[
        bool,
        float,
        int,
        str,
    ]:
        return super().peek_cfg_value(
            exp_name=exp_name,
            file_name='dec_cfg.json',
            key=key
        )

class BaseCfg(CfgMixin):
    dec_cfg_cstr = BaseDecCfg
    enc_cfg_cstr = BaseEncCfg

    def __init__(
            self,
            ckpt_step: int,
            dec_cfg: BaseDecCfg,
            enc_cfg: BaseEncCfg,
            exp_name: str,
            log_step: int,
            model_name: str,
    ):
        self.ckpt_step = ckpt_step
        self.dec_cfg = dec_cfg
        self.enc_cfg = enc_cfg
        self.exp_name = exp_name
        self.log_step = log_step
        self.model_name = model_name

    def save(self):
        super().save(exp_name=self.exp_name, file_name='exp_cfg.json')
        self.enc_cfg.save(exp_name=self.exp_name)
        self.dec_cfg.save(exp_name=self.exp_name)

    @classmethod
    def load(cls, exp_name: str) -> 'BaseCfg':
        return super().load(
            exp_name=exp_name,
            file_name='exp_cfg.json',
            enc_cfg=cls.enc_cfg_cstr.load(exp_name=exp_name),
            dec_cfg=cls.dec_cfg_cstr.load(exp_name=exp_name),
        )

    @classmethod
    def peek_cfg_value(cls, exp_name: str, key: str) -> Union[
        bool,
        float,
        int,
        str,
    ]:
        return super().peek_cfg_value(
            exp_name=exp_name,
            file_name='exp_cfg.json',
            key=key
        )
