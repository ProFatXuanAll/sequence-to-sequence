from typing import Dict

import torch

from s2s.model._rnn_attention import AttnRNNDecModel, AttnRNNEncModel, AttnRNNModel


class AttnLSTMEncModel(AttnRNNEncModel):
    def __init__(self, enc_tknzr_cfg: Dict, model_cfg: Dict):
        super().__init__(enc_tknzr_cfg=enc_tknzr_cfg, model_cfg=model_cfg)
        self.hid = torch.nn.LSTM(
            input_size=model_cfg['enc_d_hid'],
            hidden_size=model_cfg['enc_d_hid'],
            num_layers=model_cfg['enc_n_layer'],
            batch_first=True,
            dropout=model_cfg['enc_dropout'] * min(1, model_cfg['enc_n_layer'] - 1),
            bidirectional=model_cfg['is_bidir'],
        )


class AttnLSTMDecModel(AttnRNNDecModel):
    def __init__(self, dec_tknzr_cfg: Dict, model_cfg: Dict):
        super().__init__(dec_tknzr_cfg=dec_tknzr_cfg, model_cfg=model_cfg)
        self.hid = torch.nn.ModuleList(
            [
                torch.nn.LSTM(
                    input_size=model_cfg['dec_d_hid'],
                    hidden_size=model_cfg['dec_d_hid'],
                    num_layers=1,
                    batch_first=True,
                    dropout=model_cfg['dec_dropout'] * min(1, model_cfg['dec_n_layer'] - 1),
                ) for _ in range(model_cfg['dec_n_layer'])
            ]
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
                len(self.hid),
                -1,
                self.hid[0].hidden_size
        )

        # hid_tgt.shape == (B, S-1, H)
        hid_tgt = self.emb_to_hid(self.emb(tgt))

        for RNN_index in range(len(self.hid)):
            # Initialize RNN hidden state.
            # last_dec_hidden.shape == (1, B, H)
            last_dec_hidden = last_enc_hidden[RNN_index].unsqueeze(dim=0)
            
            # Initialize cell state.
            # cell_state.shape == (1, B, H)
            cell_state = torch.zeros(
                1,
                enc_hid.size(0),
                self.hid[0].hidden_size
            ).to(tgt.device)

            # To keep RNN output for each time step.
            dec_out = None

            for i in range(hid_tgt.size(-2)):
                # Calculate attention weight.
                # score.shape == (B, 1, S)
                score = last_dec_hidden.transpose(0, 1) @ enc_out.transpose(-1, -2)
                score = torch.nn.functional.softmax(score, dim=-1)
                
                # Encoder output multiply their weight.
                # attn_enc.shape == (B, S, H)
                attn_enc = enc_out * score.transpose(-1, -2)

                # Sum each time step Encoder output.
                # attn_enc.shape == (B, H)
                attn_enc = torch.sum(attn_enc, dim=1)

                # dec_input.shape == (B, 1, H)
                dec_input = (hid_tgt[:, i, :] + attn_enc).unsqueeze(dim=1)

                # Feed into RNN.
                out, (last_dec_hidden, cell_state) = self.hid[RNN_index](
                    dec_input,
                    (
                        last_dec_hidden,
                        cell_state
                    )
                )

                if dec_out == None:
                    dec_out = out
                else:
                    dec_out = torch.cat((dec_out, out), dim=1)
            hid_tgt = dec_out

        # shape: (B, S, V)
        return self.hid_to_emb(dec_out) @ self.emb.weight.transpose(0, 1)

class AttnLSTMModel(AttnRNNModel):
    model_name = 'LSTM_attention'

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
        self.enc = AttnLSTMEncModel(
            enc_tknzr_cfg=enc_tknzr_cfg,
            model_cfg=model_cfg,
        )
        self.dec = AttnLSTMDecModel(
            dec_tknzr_cfg=dec_tknzr_cfg,
            model_cfg=model_cfg,
        )
