import abc

from typing import Sequence

import torch

from s2s.model import Model
from s2s.tknzr import Tknzr


class BaseInfr(abc.ABC):
    infr_name = 'base'

    def __init__(self, **kwargs):
        pass

    @torch.no_grad()
    @abc.abstractmethod
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
        raise NotImplementedError
