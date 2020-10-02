import argparse
import sys

from s2s.cfg import BaseTknzrCfg
from s2s.cfg.model import MODEL_CFG_OPTS
from s2s.cfg.tknzr import TKNZR_CFG_OPTS
from s2s.dset import DSET_OPTS


def load_args(script: str) -> argparse.Namespace:
    if script == 'run_train_model':
        return _args_run_train_model()
    if script == 'run_train_tknzr':
        return _args_run_train_tknzr()


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


def _args_run_train_tknzr() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog='python run_train_tknzr.py',
        description='Train tokenizer.',
    )
    subparsers = parser.add_subparsers(dest='tknzr_name')

    for tknzr_name, tknzr_cfg_cstr in TKNZR_CFG_OPTS.items():
        subparser = subparsers.add_parser(
            tknzr_name,
            help=f'Train {tknzr_name} tokenizer.',
        )
        tknzr_cfg_cstr.update_parser(parser=subparser)

    subparser = subparsers.add_parser(
        'continue',
        help=f'Continue training existed tokenizer.',
    )
    BaseTknzrCfg.cont_parser(parser=subparser)

    return parser.parse_args(sys.argv[1:])
