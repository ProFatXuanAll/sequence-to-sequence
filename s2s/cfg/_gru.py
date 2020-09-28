r"""Configuration for sequence to sequence model based on GRU."""

from s2s.cfg._base import BaseModelCfg


class GRUCfg(BaseModelCfg):
    model_name = 'GRU'
