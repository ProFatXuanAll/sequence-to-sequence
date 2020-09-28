import torch

from s2s.cfg import GRUCfg
from s2s.model._rnn import RNNDecModel, RNNEncModel, RNNModel


class GRUEncModel(RNNEncModel):
    def __init__(self, cfg: GRUCfg):
        super().__init__(cfg=cfg)
        self.hid = torch.nn.GRU(
            input_size=cfg.enc_d_hid,
            hidden_size=cfg.enc_d_hid,
            num_layers=cfg.enc_n_layer,
            batch_first=True,
            dropout=cfg.enc_dropout * min(1, cfg.enc_n_layer - 1),
            bidirectional=cfg.is_bidir,
        )


class GRUDecModel(RNNDecModel):
    def __init__(self, cfg: GRUCfg):
        super().__init__(cfg=cfg)
        self.hid = torch.nn.GRU(
            input_size=cfg.dec_d_hid,
            hidden_size=cfg.dec_d_hid,
            num_layers=cfg.dec_n_layer,
            batch_first=True,
            dropout=cfg.dec_dropout * min(1, cfg.dec_n_layer - 1),
        )


class GRUModel(RNNModel):
    def __init__(self, cfg: GRUCfg):
        super(RNNModel, self).__init__()
        self.enc = GRUEncModel(cfg=cfg)
        self.dec = GRUDecModel(cfg=cfg)
