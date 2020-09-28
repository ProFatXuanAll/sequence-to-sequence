from typing import Sequence

import torch

from s2s.cfg import RNNCfg, RNNEncCfg, RNNDecCfg


class RNNEncModel(torch.nn.Module):
    def __init__(self, cfg: RNNEncCfg):
        super().__init__()
        self.emb = torch.nn.Embedding(cfg.n_vocab, cfg.d_emb, cfg.pad_id)
        self.emb_to_hid = torch.nn.Sequential(
            torch.nn.Dropout(cfg.dropout),
            torch.nn.Linear(cfg.d_emb, cfg.d_hid),
            torch.nn.ReLU(),
            torch.nn.Dropout(cfg.dropout),
            torch.nn.Linear(cfg.d_hid, cfg.d_hid),
            torch.nn.ReLU(),
            torch.nn.Dropout(cfg.dropout),
        )
        self.hid = torch.nn.RNN(
            input_size=cfg.d_hid,
            hidden_size=cfg.d_hid,
            num_layers=cfg.n_layer,
            batch_first=True,
            dropout=cfg.dropout * min(1, cfg.n_layer - 1),
            bidirectional=cfg.is_bidir,
        )

    def forward(
            self,
            src: torch.Tensor,
            src_len: torch.Tensor,
    ) -> torch.Tensor:
        r"""Encoder forward.

        src.shape == (B, S)
        src.dtype == torch.int
        src_len.shape == (B)
        src_len.dtype == torch.int
        """
        # shape: (B, S, H * (is_bidir + 1))
        out, _ = self.hid(self.emb_to_hid(self.emb(src)))

        # shape: (B, H * (is_bidir + 1))
        return out[torch.arange(out.size(0)).to(out.device), src_len]


class RNNDecModel(torch.nn.Module):
    def __init__(self, cfg: RNNDecCfg):
        super().__init__()
        self.emb = torch.nn.Embedding(cfg.n_vocab, cfg.d_emb, cfg.pad_id)
        self.emb_to_hid = torch.nn.Sequential(
            torch.nn.Dropout(cfg.dropout),
            torch.nn.Linear(cfg.d_emb, cfg.d_hid),
            torch.nn.ReLU(),
            torch.nn.Dropout(cfg.dropout),
            torch.nn.Linear(cfg.d_hid, cfg.d_hid),
            torch.nn.ReLU(),
            torch.nn.Dropout(cfg.dropout),
        )
        self.enc_to_hid = torch.nn.Sequential(
            torch.nn.Dropout(cfg.dropout),
            torch.nn.Linear(cfg.d_enc_hid, cfg.d_hid * cfg.n_layer),
            torch.nn.ReLU(),
            torch.nn.Dropout(cfg.dropout),
            torch.nn.Linear(cfg.d_hid * cfg.n_layer, cfg.d_hid * cfg.n_layer),
            torch.nn.ReLU(),
            torch.nn.Dropout(cfg.dropout),
        )
        self.hid = torch.nn.RNN(
            input_size=cfg.d_hid,
            hidden_size=cfg.d_hid,
            num_layers=cfg.n_layer,
            batch_first=True,
            dropout=cfg.dropout * min(1, cfg.n_layer - 1),
        )
        self.hid_to_emb = torch.nn.Sequential(
            torch.nn.Dropout(cfg.dropout),
            torch.nn.Linear(cfg.d_hid, cfg.d_emb),
            torch.nn.ReLU(),
            torch.nn.Dropout(cfg.dropout),
            torch.nn.Linear(cfg.d_emb, cfg.d_emb),
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
            self.enc_to_hid(enc_hid)
        )

        # shape: (B, S, E)
        return self.hid_to_emb(out)


class RNNModel(torch.nn.Module):
    def __init__(self, cfg: RNNCfg):
        super().__init__()
        self.enc = RNNEncModel(cfg=cfg.enc_cfg)
        self.dec = RNNDecModel(cfg=cfg.dec_cfg)

    def forward(
            self,
            src: torch.Tensor,
            src_len: torch.Tensor,
            tgt: torch.Tensor,
    ) -> torch.Tensor:
        # shape: (B, S, E)
        return self.dec(self.enc(src), tgt)

    def predict(
            self,
            src: torch.Tensor,
            src_len: torch.Tensor,
            tgt: torch.Tensor,
            tgt_len: torch.Tensor,
    ):
        # shape: (B, S, E)
        logits = self(src=src, src_len=src_len, tgt=tgt)

        # shape: (B)
        return logits[
            torch.arange(logits.size(0)).to(logits.device),
            tgt_len
        ].argmax(dim=-1)
