import argparse
import os
import re

from typing import Dict
from typing import Type

import s2s.cfg
import s2s.model
import s2s.path


def load_model(
        args: argparse.Namespace,
        cfg: s2s.cfg.BaseCfg
) -> s2s.model.Model:
    model = s2s.model.MODEL_OPTS[cfg.model_name](cfg=cfg)

    if args.ckpt != 0:
        exp_path = os.path.join(s2s.path.EXP_PATH, cfg.exp_name)

        if args.ckpt == -1:
            prog = re.compile(r'model-(\d+).pt')
            file_names = filter(
                lambda file_name: prog.match(file_name),
                os.listdir(exp_path)
            )
            args.ckpt = max(map(
                lambda file_name: prog.match(file_name)[0],
                file_names)
            )

        file_path = os.path.join(exp_path, f'model-{args.ckpt}.pt')
        model.load_state_dict(torch.load(file_path))

    return model
