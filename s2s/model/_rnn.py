import torch

from s2s.cfg import RNNCfg


class RNNEncModel(torch.nn.Module):
    def __init__(self, cfg: RNNCfg):
        super().__init__()
        self.emb = torch.nn.Embedding(
            num_embeddings=cfg.enc_n_vocab,
            embedding_dim=cfg.enc_d_emb,
            padding_idx=cfg.enc_pad_id,
        )
        self.emb_to_hid = torch.nn.Sequential(
            torch.nn.Dropout(
                p=cfg.enc_dropout,
            ),
            torch.nn.Linear(
                in_features=cfg.enc_d_emb,
                out_features=cfg.enc_d_hid,
            ),
            torch.nn.ReLU(),
            torch.nn.Dropout(
                p=cfg.enc_dropout,
            ),
            torch.nn.Linear(
                in_features=cfg.enc_d_hid,
                out_features=cfg.enc_d_hid,
            ),
            torch.nn.ReLU(),
            torch.nn.Dropout(
                p=cfg.enc_dropout,
            ),
        )
        self.hid = torch.nn.RNN(
            input_size=cfg.enc_d_hid,
            hidden_size=cfg.enc_d_hid,
            num_layers=cfg.enc_n_layer,
            batch_first=True,
            dropout=cfg.enc_dropout * min(1, cfg.enc_n_layer - 1),
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

    def _forward_unimplemented(self):
        pass

class RNNDecModel(torch.nn.Module):
    def __init__(self, cfg: RNNCfg):
        super().__init__()
        self.emb = torch.nn.Embedding(
            num_embeddings=cfg.dec_n_vocab,
            embedding_dim=cfg.dec_d_emb,
            padding_idx=cfg.dec_pad_id,
        )
        self.emb_to_hid = torch.nn.Sequential(
            torch.nn.Dropout(
                p=cfg.dec_dropout,
            ),
            torch.nn.Linear(
                in_features=cfg.dec_d_emb,
                out_features=cfg.dec_d_hid,
            ),
            torch.nn.ReLU(),
            torch.nn.Dropout(
                p=cfg.dec_dropout,
            ),
            torch.nn.Linear(
                in_features=cfg.dec_d_hid,
                out_features=cfg.dec_d_hid,
            ),
            torch.nn.ReLU(),
            torch.nn.Dropout(
                p=cfg.dec_dropout,
            ),
        )
        self.enc_to_hid = torch.nn.Sequential(
            torch.nn.Dropout(
                p=cfg.dec_dropout,
            ),
            torch.nn.Linear(
                in_features=cfg.enc_d_hid,
                out_features=cfg.dec_d_hid * cfg.dec_n_layer,
            ),
            torch.nn.ReLU(),
            torch.nn.Dropout(
                p=cfg.dec_dropout,
            ),
            torch.nn.Linear(
                in_features=cfg.dec_d_hid * cfg.dec_n_layer,
                out_features=cfg.dec_d_hid * cfg.dec_n_layer,
            ),
            torch.nn.ReLU(),
            torch.nn.Dropout(
                p=cfg.dec_dropout,
            ),
        )
        self.hid = torch.nn.RNN(
            input_size=cfg.dec_d_hid,
            hidden_size=cfg.dec_d_hid,
            num_layers=cfg.dec_n_layer,
            batch_first=True,
            dropout=cfg.dec_dropout * min(1, cfg.dec_n_layer - 1),
        )
        self.hid_to_emb = torch.nn.Sequential(
            torch.nn.Dropout(
                p=cfg.dec_dropout,
            ),
            torch.nn.Linear(
                in_features=cfg.dec_d_hid,
                out_features=cfg.dec_d_emb,
            ),
            torch.nn.ReLU(),
            torch.nn.Dropout(
                p=cfg.dec_dropout,
            ),
            torch.nn.Linear(
                in_features=cfg.dec_d_emb,
                out_features=cfg.dec_d_emb
            ),
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

    def _forward_unimplemented(self):
        pass

class RNNModel(torch.nn.Module):
    def __init__(self, cfg: RNNCfg):
        super().__init__()
        self.enc = RNNEncModel(cfg=cfg)
        self.dec = RNNDecModel(cfg=cfg)

    def forward(
            self,
            src: torch.Tensor,
            src_len: torch.Tensor,
            tgt: torch.Tensor,
    ) -> torch.Tensor:
        # shape: (B, S, E)
        return self.dec(self.enc(src=src, src_len=src_len), tgt=tgt)

    def _forward_unimplemented(self):
        pass

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
