import argparse
import sys

from typing import Dict
from typing import Type

import s2s.cfg


def load_cfg(args: argparse.Namespace) -> s2s.cfg.BaseCfg:
    if args.ckpt == 0:
        return s2s.cfg.CFG_OPTS[args.model_name].load_from_args(args=args)

    model_name = s2s.cfg.BaseCfg.peek_cfg_value(
        exp_name=args.exp_name,
        key='model_name',
    )

    return s2s.cfg.CFG_OPTS[model_name].load(exp_name=args.exp_name)
