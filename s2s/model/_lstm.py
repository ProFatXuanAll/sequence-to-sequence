import argparse
import torch

from s2s.model._rnn import RNNDecModel, RNNEncModel, RNNModel


class LSTMEncModel(RNNEncModel):
    def __init__(
        self,
        enc_tknzr_cfg: argparse.Namespace,
        model_cfg: argparse.Namespace
    ):
        super().__init__(enc_tknzr_cfg=enc_tknzr_cfg, model_cfg=model_cfg)
        self.hid = torch.nn.LSTM(
            input_size=model_cfg.enc_d_hid,
            hidden_size=model_cfg.enc_d_hid,
            num_layers=model_cfg.enc_n_layer,
            batch_first=True,
            dropout=model_cfg.enc_dropout * min(1, model_cfg.enc_n_layer - 1),
            bidirectional=model_cfg.is_bidir,
        )


class LSTMDecModel(RNNDecModel):
    def __init__(
        self,
        dec_tknzr_cfg: argparse.Namespace,
        model_cfg: argparse.Namespace
    ):
        super().__init__(dec_tknzr_cfg=dec_tknzr_cfg, model_cfg=model_cfg)
        self.hid = torch.nn.LSTM(
            input_size=model_cfg.dec_d_hid,
            hidden_size=model_cfg.dec_d_hid,
            num_layers=model_cfg.dec_n_layer,
            batch_first=True,
            dropout=model_cfg.dec_dropout * min(1, model_cfg.dec_n_layer - 1),
        )

    def forward(
            self,
            enc_hid: torch.Tensor,
            tgt: torch.Tensor,
    ) -> torch.Tensor:
        r"""Decoder forward.

        enc_hid.shape == (B, H)
        enc_hid.dtype == torch.float
        tgt.shape == (B, S)
        tgt.dtype == torch.int
        """
        # shape: (B, S, H)
        out, _ = self.hid(
            self.emb_to_hid(self.emb(tgt)),
            (
                self.enc_to_hid(enc_hid).reshape(
                    self.hid.num_layers,
                    -1,
                    self.hid.hidden_size
                ),
                torch.zeros(
                    self.hid.num_layers,
                    enc_hid.size(0),
                    self.hid.hidden_size
                ).to(tgt.device),
            )
        )

        # shape: (B, S, V)
        return self.hid_to_emb(out) @ self.emb.weight.transpose(0, 1)


class LSTMModel(RNNModel):
    model_name = 'LSTM'

    def __init__(
            self,
            dec_tknzr_cfg: argparse.Namespace,
            enc_tknzr_cfg: argparse.Namespace,
            model_cfg: argparse.Namespace,
    ):
        super().__init__(
            dec_tknzr_cfg=dec_tknzr_cfg,
            enc_tknzr_cfg=enc_tknzr_cfg,
            model_cfg=model_cfg,
        )
        self.enc = LSTMEncModel(
            enc_tknzr_cfg=enc_tknzr_cfg,
            model_cfg=model_cfg,
        )
        self.dec = LSTMDecModel(
            dec_tknzr_cfg=dec_tknzr_cfg,
            model_cfg=model_cfg,
        )
