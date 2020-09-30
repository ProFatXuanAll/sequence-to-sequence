import argparse
import sys

import s2s


def load_args(script: str) -> argparse.Namespace:
    if script == 'run_train_model':
        return _args_run_train_model()
    if script == 'run_train_tker':
        return _args_run_train_tker()


def _args_run_train_tker() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog='python run_train_tker.py',
        description='Train tokenizer.',
    )
    subparsers = parser.add_subparsers(dest='tker_name')

    for tker_name, tker_cfg_cstr in s2s.cfg.tker.TKER_CFG_OPTS.items():
        subparser = subparsers.add_parser(
            tker_name,
            help=f'Train {tker_name} tokenizer.',
        )
        parser.add_argument(
            '--dset_name',
            help='',
            required=True,
            type=str,
        )
        s2s.cfg.BaseExpCfg.update_parser(parser=subparser)
        tker_cfg_cstr.update_parser(parser=subparser)

    return parser.parse_args(sys.argv[1:])

def _args_run_train_model() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog='python run_train_model.py',
        description='Train sequence-to-sequence model.',
    )
    subparsers = parser.add_subparsers(dest='model_name')

    for model_name, model_cfg_cstr in s2s.cfg.model.MODEL_CFG_OPTS.items():
        subparser = subparsers.add_parser(
            model_name,
            help=f'Train {model_name} sequence-to-sequence model.',
        )
        s2s.cfg.BaseExpCfg.update_parser(parser=subparser)
        subparser.add_argument(
            '--ckpt',
            default=0,
            help='Start from specific checkpoint.',
            type=int,
        )
        subparser.add_argument(
            '--ckpt_step',
            help='Checkpoint save interval.',
            required=True,
            type=int,
        )
        parser.add_argument(
            '--dset_name',
            help='',
            required=True,
            type=str,
        )
        subparser.add_argument(
            '--log_step',
            help='Performance log interval.',
            required=True,
            type=int,
        )
        model_cfg_cstr.update_parser(parser=subparser)

    return parser.parse_args(sys.argv[1:])
