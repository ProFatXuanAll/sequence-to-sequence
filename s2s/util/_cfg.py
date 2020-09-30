import argparse
import sys

from typing import Dict
from typing import Type

import s2s.cfg


def load_model_cfg(args: argparse.Namespace) -> s2s.cfg.BaseModelCfg:
    if args.ckpt == 0:
        return s2s.cfg.model.MODEL_CFG_OPTS[args.model_name](**args.__dict__)

    model_name = s2s.cfg.BaseModelCfg(**args.__dict__).model_name

    return s2s.cfg.model.MODEL_CFG_OPTS[model_name].load(
        exp_name=args.exp_name
    )

def load_tker_cfg(args: argparse.Namespace) -> s2s.cfg.BaseTkerCfg:
    # When perform `run_train_tker.py`.
    if 'tker_name' in args.__dict__:
        return s2s.cfg.tker.TKER_CFG_OPTS[args.tker_name](**args.__dict__)
    if 'dec_tker_exp' in args.__dict__:
        return s2s.cfg.BaseTkerCfg.load(exp_name=args.dec_tker_exp)
    if 'enc_tker_exp' in args.__dict__:
        return s2s.cfg.BaseTkerCfg.load(exp_name=args.enc_tker_exp)
