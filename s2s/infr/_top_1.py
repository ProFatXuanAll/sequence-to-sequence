import torch

from typing import Sequence

from s2s.infr._base import BaseInfr
from s2s.model import Model
from s2s.tknzr import Tknzr


class Top1Infr(BaseInfr):
    infr_name = 'top_1'

    @torch.no_grad()
    def gen(
        self,
        batch_text: Sequence[str],
        dec_max_len: int,
        dec_tknzr: Tknzr,
        device: torch.device,
        enc_max_len: int,
        enc_tknzr: Tknzr,
        model: Model,
        **kwargs,
    ) -> Sequence[str]:

        src, src_len = enc_tknzr.batch_enc(
            batch_text=batch_text,
            max_len=enc_max_len,
        )
        tgt, tgt_len = dec_tknzr.batch_enc(
            batch_text=[''] * len(src_len),
            max_len=min(2, dec_max_len),
        )

        src = torch.tensor(src).to(device)
        src_len = torch.tensor(src_len).to(device)
        tgt = torch.tensor(tgt).to(device)
        tgt_len = torch.tensor(tgt_len).to(device)
        tgt = tgt[:, :-1]
        tgt_len -= 1

        for _ in range(dec_max_len):
            # shape: (B, S, V)
            logits = model(
                src=src,
                src_len=src_len,
                tgt=tgt,
                tgt_len=tgt_len,
            )
            # shape: (B, 1)
            pred = logits[:, -1, :].argmax(dim=-1).reshape(-1, 1)

            # Update target sequence.
            tgt = torch.cat([tgt, pred], dim=-1)
            tgt_len += 1

        return dec_tknzr.batch_dec(batch_tk_ids=tgt.tolist())
