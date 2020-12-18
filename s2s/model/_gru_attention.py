from typing import Dict

import torch

from s2s.model._rnn_attention import AttnRNNDecModel, AttnRNNEncModel, AttnRNNModel, AttnRNNBlock

class AttnGRUBlock(AttnRNNBlock):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__(input_size, hidden_size, num_layers)
        self.hid = torch.nn.ModuleList(
            [
                torch.nn.GRU(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=1,
                    batch_first=True,
                ) for _ in range(num_layers)
            ]
        )

class AttnGRUEncModel(AttnRNNEncModel):
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

class AttnGRUDecModel(AttnRNNDecModel):
    def __init__(self, dec_tknzr_cfg: Dict, model_cfg: Dict):
        super().__init__(dec_tknzr_cfg=dec_tknzr_cfg, model_cfg=model_cfg)
        self.hid = AttnGRUBlock(
            input_size=model_cfg['dec_d_hid'],
            hidden_size=model_cfg['dec_d_hid'],
            num_layers=model_cfg['dec_n_layer']
        )

class AttnGRUModel(AttnRNNModel):
    model_name = 'GRU_attention'

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
        self.enc = AttnGRUEncModel(
            enc_tknzr_cfg=enc_tknzr_cfg,
            model_cfg=model_cfg,
        )
        self.dec = AttnGRUDecModel(
            dec_tknzr_cfg=dec_tknzr_cfg,
            model_cfg=model_cfg,
        )
