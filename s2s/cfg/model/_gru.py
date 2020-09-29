r"""Configuration for sequence to sequence model based on GRU."""

from s2s.cfg.model._rnn import RNNModelCfg


class GRUModelCfg(RNNModelCfg):
    model_name = 'GRU'
