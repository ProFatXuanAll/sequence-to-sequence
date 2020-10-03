from typing import Dict

import torch

from s2s.model._rnn import RNNDecModel, RNNEncModel, RNNModel


class GRUEncModel(RNNEncModel):
    def __init__(self, enc_tknzr_cfg: Dict, model_cfg: Dict):
        super().__init__(enc_tknzr_cfg=enc_tknzr_cfg, model_cfg=model_cfg)
        self.hid = torch.nn.GRU(
            input_size=model_cfg['enc_d_hid'],
            hidden_size=model_cfg['enc_d_hid'],
            num_layers=model_cfg['enc_n_layer'],
            batch_first=True,
            dropout=model_cfg['enc_dropout'] * min(1, model_cfg['enc_n_layer'] - 1),
            bidirectional=model_cfg['is_bidir'],
        )


class GRUDecModel(RNNDecModel):
    def __init__(self, dec_tknzr_cfg: Dict, model_cfg: Dict):
        super().__init__(dec_tknzr_cfg=dec_tknzr_cfg, model_cfg=model_cfg)
        self.hid = torch.nn.GRU(
            input_size=model_cfg['dec_d_hid'],
            hidden_size=model_cfg['dec_d_hid'],
            num_layers=model_cfg['dec_n_layer'],
            batch_first=True,
            dropout=model_cfg['dec_dropout'] * min(1, model_cfg['dec_n_layer'] - 1),
        )


class GRUModel(RNNModel):
    model_name = 'GRU'

    def __init__(
            self,
            dec_tknzr_cfg: Dict,
            enc_tknzr_cfg: Dict,
            model_cfg: Dict,
    ):
        super().__init__(
            dec_tknzr_cfg=dec_tknzr_cfg,
            enc_tknzr_cfg=enc_tknzr_cfg,
            model_cfg=model_cfg,
        )
        self.enc = GRUEncModel(
            enc_tknzr_cfg=enc_tknzr_cfg,
            model_cfg=model_cfg,
        )
        self.dec = GRUDecModel(
            dec_tknzr_cfg=dec_tknzr_cfg,
            model_cfg=model_cfg,
        )
