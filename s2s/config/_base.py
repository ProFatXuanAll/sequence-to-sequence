r"""Configuration for sequence to sequence model."""

import json
import os

import s2s.path

class CfgMixin:
    def __iter__(self):
        for key, value in self.__dict__.items()
            yield key, value

    def save(self, exp_name: str, file_name: str) -> None:
        cfg = {k: v for k, v in iter(self)}
        exp_path = os.path.join(s2s.path.EXP_PATH, exp_name)
        file_path = os.path.join(exp_path, file_name)

        if not os.path.exists(exp_path):
            os.makedirs(exp_path)

        with open(file_path, encoding='utf-8', 'w') as cfg_file:
            json.dump(cfg, cfg_file, ensure_ascii=False, indent=2)

class BaseEncCfg(CfgMixin):
    def save(self, exp_name: str) -> None:
        super().save(exp_name=exp_name, file_name='enc_cfg.json')


class BaseDecCfg(CfgMixin):
    def save(self, exp_name: str) -> None:
        super().save(exp_name=exp_name, file_name='dec_cfg.json')


class BaseCfg(CfgMixin):
    def __init__(
            self,
            ckpt_step: int,
            exp_name: str,
            log_step: int,
    ):
        self.ckpt_step = ckpt_step
        self.exp_name = exp_name
        self.log_step = log_step
        self.dec_cfg = BaseDecCfg()
        self.enc_cfg = BaseEncCfg()

    def save(self):
        cfg = {}
        for k, v in self:
            if isinstance(v, (BaseEncCfg, BaseDecCfg)):
                v.save(self.exp_name)
                continue
            cfg[k] = v

        exp_path = os.path.join(s2s.path.EXP_PATH, self.exp_name)
        file_path = os.path.join(exp_path, 'exp.json')

        if not os.path.exists(exp_path):
            os.makedirs(exp_path)

        with open(file_path, encoding='utf-8', 'w') as cfg_file:
            json.dump(cfg, cfg_file, ensure_ascii=False, indent=2)
