import argparse

import torch

class AttnRNNBlock(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.input_size = input_size

        self.hid = torch.nn.ModuleList(
            [
                torch.nn.RNN(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=1,
                    batch_first=True,
                ) for _ in range(num_layers)
            ]
        )
        self.W = torch.nn.Linear(
            in_features=hidden_size,
            out_features=hidden_size
        )
    def get_num_layers(self):
        return self.num_layers

    def get_hidden_size(self):
        return self.hidden_size

    def forward(self, enc_out, first_hid_tgt, last_enc_hidden):
        r"""AttnRNNBlock forward

        first_hid_tgt.shape == (B, S-1, H)
        first_hid_tgt.dtype == torch.float
        last_enc_hidden.shape == (num_layer, B, H)
        last_enc_hidden.dtype == torch.float
        enc_out.shape == (B, S, H)
        enc_out.dtype == torch.float
        """

        hid_tgt = first_hid_tgt

        for RNN_index in range(len(self.hid)):
            # Initialize RNN hidden state.
            # dec_hidden.shape == (1, B, H)
            dec_hidden = last_enc_hidden[RNN_index].unsqueeze(dim=0)

            # To keep RNN output for each time step.
            dec_out = None

            for i in range(hid_tgt.size(-2)):
                # Calculate attention weight.
                # score.shape == (B, 1, S)
                score = dec_hidden.transpose(0, 1) @ enc_out.transpose(-1, -2)
                score = torch.nn.functional.softmax(score, dim=-1)

                # Encoder output multiply their weight.
                # attn_enc.shape == (B, S, H)
                attn_enc = enc_out * score.transpose(-1, -2)

                # Sum each time step Encoder output.
                # attn_enc.shape == (B, H)
                attn_enc = torch.sum(attn_enc, dim=1)

                # dec_input.shape == (B, 1, H)
                dec_input = (hid_tgt[:, i, :] + self.W(attn_enc)).unsqueeze(dim=1)

                # Feed into RNN.
                out, dec_hidden = self.hid[RNN_index](dec_input, dec_hidden)

                if dec_out == None:
                    dec_out = out
                else:
                    dec_out = torch.cat((dec_out, out), dim=1)

            hid_tgt = dec_out
        return dec_out

class AttnRNNEncModel(torch.nn.Module):
    def __init__(
        self,
        enc_tknzr_cfg: argparse.Namespace,
        model_cfg: argparse.Namespace
        ):
        super().__init__()
        self.emb = torch.nn.Embedding(
            num_embeddings=enc_tknzr_cfg.n_vocab,
            embedding_dim=model_cfg.enc_d_emb,
            padding_idx=0,
        )
        self.emb_to_hid = torch.nn.Sequential(
            torch.nn.Dropout(
                p=model_cfg.enc_dropout,
            ),
            torch.nn.Linear(
                in_features=model_cfg.enc_d_emb,
                out_features=model_cfg.enc_d_hid,
            ),
            torch.nn.ReLU(),
            torch.nn.Dropout(
                p=model_cfg.enc_dropout,
            ),
            torch.nn.Linear(
                in_features=model_cfg.enc_d_hid,
                out_features=model_cfg.enc_d_hid,
            ),
            torch.nn.ReLU(),
            torch.nn.Dropout(
                p=model_cfg.enc_dropout,
            ),
        )
        self.hid = torch.nn.RNN(
            input_size=model_cfg.enc_d_hid,
            hidden_size=model_cfg.enc_d_hid,
            num_layers=model_cfg.enc_n_layer,
            batch_first=True,
            dropout=model_cfg.enc_dropout * min(1, model_cfg.enc_n_layer - 1),
            bidirectional=model_cfg.is_bidir,
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
        return out[torch.arange(out.size(0)).to(out.device), src_len - 1], out


class AttnRNNDecModel(torch.nn.Module):
    def __init__(
        self,
        dec_tknzr_cfg: argparse.Namespace,
        model_cfg: argparse.Namespace
    ):
        super().__init__()
        self.emb = torch.nn.Embedding(
            num_embeddings=dec_tknzr_cfg.n_vocab,
            embedding_dim=model_cfg.dec_d_emb,
            padding_idx=0,
        )
        self.emb_to_hid = torch.nn.Sequential(
            torch.nn.Dropout(
                p=model_cfg.dec_dropout,
            ),
            torch.nn.Linear(
                in_features=model_cfg.dec_d_emb,
                out_features=model_cfg.dec_d_hid,
            ),
            torch.nn.ReLU(),
            torch.nn.Dropout(
                p=model_cfg.dec_dropout,
            ),
            torch.nn.Linear(
                in_features=model_cfg.dec_d_hid,
                out_features=model_cfg.dec_d_hid,
            ),
            torch.nn.ReLU(),
            torch.nn.Dropout(
                p=model_cfg.dec_dropout,
            ),
        )
        self.enc_to_hid = torch.nn.Sequential(
            torch.nn.Dropout(
                p=model_cfg.dec_dropout,
            ),
            torch.nn.Linear(
                in_features=model_cfg.enc_d_hid * (model_cfg.is_bidir + 1),
                out_features=model_cfg.dec_d_hid * model_cfg.dec_n_layer,
            ),
            torch.nn.ReLU(),
            torch.nn.Dropout(
                p=model_cfg.dec_dropout,
            ),
            torch.nn.Linear(
                in_features=model_cfg.dec_d_hid * model_cfg.dec_n_layer,
                out_features=model_cfg.dec_d_hid * model_cfg.dec_n_layer,
            ),
            torch.nn.ReLU(),
            torch.nn.Dropout(
                p=model_cfg.dec_dropout,
            ),
        )
        self.hid = AttnRNNBlock(
            input_size=model_cfg.dec_d_hid,
            hidden_size=model_cfg.dec_d_hid,
            num_layers=model_cfg.dec_n_layer
        )
        self.hid_to_emb = torch.nn.Sequential(
            torch.nn.Dropout(
                p=model_cfg.dec_dropout,
            ),
            torch.nn.Linear(
                in_features=model_cfg.dec_d_hid,
                out_features=model_cfg.dec_d_hid,
            ),
            torch.nn.ReLU(),
            torch.nn.Dropout(
                p=model_cfg.dec_dropout,
            ),
            torch.nn.Linear(
                in_features=model_cfg.dec_d_hid,
                out_features=model_cfg.dec_d_emb,
            ),
        )

    def forward(
            self,
            enc_hid: torch.Tensor,
            enc_out: torch.Tensor,
            tgt: torch.Tensor,
    ) -> torch.Tensor:
        r"""Decoder forward.

        enc_hid.shape == (B, H)
        enc_hid.dtype == torch.float
        enc_out.shape == (B, S, H)
        enc_out.dtype == torch.float
        tgt.shape == (B, S)
        tgt.dtype == torch.int
        """
        # last_enc_hidden.shape == (num_layer, B, H)
        last_enc_hidden = self.enc_to_hid(enc_hid).reshape(
                self.hid.get_num_layers(),
                -1,
                self.hid.get_hidden_size()
        )

        # hid_tgt.shape == (B, S-1, H)
        hid_tgt = self.emb_to_hid(self.emb(tgt))

        # dec_out.shape == (B, S-1, H)
        dec_out = self.hid(
            enc_out=enc_out,
            first_hid_tgt=hid_tgt,
            last_enc_hidden=last_enc_hidden
        )

        # shape: (B, S-1, V)
        return self.hid_to_emb(dec_out) @ self.emb.weight.transpose(0, 1)

class AttnRNNModel(torch.nn.Module):
    model_name = 'RNN_attention'

    def __init__(
            self,
            dec_tknzr_cfg: argparse.Namespace,
            enc_tknzr_cfg: argparse.Namespace,
            model_cfg: argparse.Namespace,
    ):
        super().__init__()
        self.enc = AttnRNNEncModel(
            enc_tknzr_cfg=enc_tknzr_cfg,
            model_cfg=model_cfg,
        )
        self.dec = AttnRNNDecModel(
            dec_tknzr_cfg=dec_tknzr_cfg,
            model_cfg=model_cfg,
        )

    def forward(
            self,
            src: torch.Tensor,
            src_len: torch.Tensor,
            tgt: torch.Tensor,
            **kwargs,
    ) -> torch.Tensor:
        # shape: (B, S, V)
        enc_hid, enc_out = self.enc(src=src, src_len=src_len)
        return self.dec(enc_hid, enc_out, tgt=tgt)

    @classmethod
    def update_subparser(cls, subparser: argparse.ArgumentParser):
        subparser.add_argument(
            '--dec_d_emb',
            help='Decoder embedding dimension.',
            required=True,
            type=int,
        )
        subparser.add_argument(
            '--dec_d_hid',
            help='Decoder hidden dimension.',
            required=True,
            type=int,
        )
        subparser.add_argument(
            '--dec_dropout',
            help='Decoder dropout rate.',
            required=True,
            type=float,
        )
        subparser.add_argument(
            '--dec_n_layer',
            help=f'Number of decoder {cls.model_name} layer(s).',
            required=True,
            type=int,
        )
        subparser.add_argument(
            '--enc_d_emb',
            help='Encoder embedding dimension.',
            required=True,
            type=int,
        )
        subparser.add_argument(
            '--enc_d_hid',
            help='Encoder hidden dimension.',
            required=True,
            type=int,
        )
        subparser.add_argument(
            '--enc_dropout',
            help='Encoder dropout rate.',
            required=True,
            type=float,
        )
        subparser.add_argument(
            '--enc_n_layer',
            help=f'Number of encoder {cls.model_name} layer(s).',
            required=True,
            type=int,
        )
        subparser.add_argument(
            '--is_bidir',
            help=f'Whether to use bidirectional {cls.model_name} encoder or not.',
            action='store_true',
        )
