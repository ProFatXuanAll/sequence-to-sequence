import argparse

from typing import Dict
from typing import Type

import s2s.cfg

_CFG_MAP: Dict[str, Type[s2s.cfg.BaseCfg]] = {
    'gru': s2s.cfg.GRUCfg,
    'rnn': s2s.cfg.RNNCfg,
}

def load_cfg(args: argparse.Namespace) -> s2s.cfg.BaseCfg:
    if args.ckpt == 0:
        return _CFG_MAP[args.model_name].parse_args(args=args)

    model_name = s2s.cfg.BaseCfg.peek_cfg_value(
        exp_name=args.exp_name,
        key='model_name'
    )

    return _CFG_MAP[model_name].load(exp_name=args.exp_name)
