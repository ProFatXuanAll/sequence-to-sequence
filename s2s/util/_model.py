import os
import re

import torch

from s2s.model import Model
from s2s.path import EXP_PATH


def load_model_from_ckpt(ckpt: int, exp_name: str, model: Model) -> Model:
    exp_path = os.path.join(EXP_PATH, exp_name)
    prog = re.compile(r'model-(\d+).pt')
    all_ckpt = list(filter(
        lambda file_name: prog.match(file_name),
        os.listdir(exp_path),
    ))

    # Load last checkpoint.
    if ckpt == -1:
        ckpt = max(map(
            lambda file_name: int(prog.match(file_name)[1]),
            all_ckpt,
        ))

    file_path = os.path.join(exp_path, f'model-{ckpt}.pt')

    if not os.path.exists(file_path):
        raise FileNotFoundError(f'{file_path} does not exist.')

    model.load_state_dict(torch.load(file_path))
    return model
