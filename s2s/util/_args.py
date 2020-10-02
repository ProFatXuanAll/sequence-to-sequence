import argparse
import sys

from s2s.cfg import BaseTknzrCfg
from s2s.cfg.model import MODEL_CFG_OPTS
from s2s.dset import DSET_OPTS


def load_args(script: str) -> argparse.Namespace:
    if script == 'run_train_model':
        return _args_run_train_model()


def _args_run_train_model() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog='python run_train_model.py',
        description='Train sequence-to-sequence model.',
    )
    subparsers = parser.add_subparsers(dest='model_name')

    for model_name, model_cfg_cstr in MODEL_CFG_OPTS.items():
        subparser = subparsers.add_parser(
            model_name,
            help=f'Train {model_name} sequence-to-sequence model.',
        )
        model_cfg_cstr.update_parser(parser=subparser)

    return parser.parse_args(sys.argv[1:])
