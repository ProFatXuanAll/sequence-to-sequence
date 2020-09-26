import torch

from s2s.config import RNNCfg, RNNEncCfg, RNNDecCfg

class RNNEncModel(torch.nn.Module):
    def __init__(self, cfg: RNNEncCfg):
        super().__init__()
        self.emb = torch.nn.Embedding(
            num_embeddings=cfg.n_vocab,
            embedding_dim=cfg.d_emb,
            padding_idx=cfg.pad_id,
        )
        self.emb_to_hid = torch.nn.Sequential(
            torch.nn.Dropout(p=cfg.dropout),
            torch.nn.Linear(in_features=cfg.d_emb, out_features=cfg.d_hid),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=cfg.dropout),
            torch.nn.Linear(in_features=cfg.d_hid, out_features=cfg.d_hid),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=cfg.dropout),
        )
        self.hid = torch.nn.RNN(
            input_size=cfg.d_hid,
            hidden_size=cfg.d_hid,
            num_layers=cfg.n_layer,
            batch_first=True,
            dropout=cfg.dropout * min(1, cfg.n_layer - 1),
            bidirectional=cfg.is_bidir,
        )

    def forward(self, src) -> torch.Tensor:
        # shape: (B, H * (is_bidir + 1))
        _, h_t = self.hid(self.emb_to_hid(self.emb(src)))
        return h_t


class RNNDecModel(torch.nn.Module):
    def __init__(self, cfg: RNNDecCfg):
        super().__init__()
        self.emb = torch.nn.Embedding(
            num_embeddings=cfg.n_vocab,
            embedding_dim=cfg.d_emb,
            padding_idx=cfg.pad_id,
        )
        self.emb_to_hid = torch.nn.Sequential(
            torch.nn.Dropout(p=cfg.dropout),
            torch.nn.Linear(in_features=cfg.d_emb, out_features=cfg.d_hid),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=cfg.dropout),
            torch.nn.Linear(in_features=cfg.d_hid, out_features=cfg.d_hid),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=cfg.dropout),
        )
        self.enc_to_hid = torch.nn.Sequential(
            torch.nn.Dropout(p=cfg.dropout),
            torch.nn.Linear(in_features=cfg.d_enc_hid, out_features=cfg.d_hid * cfg.n_layer),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=cfg.dropout),
            torch.nn.Linear(in_features=cfg.d_hid, out_features=cfg.d_hid * cfg.n_layer),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=cfg.dropout),
        )
        self.hid = torch.nn.RNN(
            input_size=cfg.d_hid,
            hidden_size=cfg.d_hid,
            num_layers=cfg.n_layer,
            batch_first=True,
            dropout=cfg.dropout * min(1, cfg.n_layer - 1),
        )
        self.hid_to_emd = torch.nn.Sequential(
            torch.nn.Dropout(p=cfg.dropout),
            torch.nn.Linear(in_features=cfg.d_hid, out_features=cfg.d_emd),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=cfg.dropout),
            torch.nn.Linear(in_features=cfg.d_emd, out_features=cfg.d_emd),
        )

    def forward(self, enc_hid, tgt) -> torch.Tensor:
        # shape: (B, S, H)
        out, _ = self.hid(self.emb_to_hid(self.emb(tgt)), self.enc_to_hid(enc_hid))

        # shape: (B, S, E)
        return self.hid_to_emd(out)


class RNNModel(torch.nn.Module):
    def __init__(self, cfg: RNNCfg):
        super().__init__()
        self.enc = RNNEncModel(cfg=cfg.enc_cfg)
        self.dec = RNNDecModel(cfg=cfg.dec_cfg)

    def forward(self, src, tgt):
        return self.dec(self.enc(src), tgt)
