import argparse
import sys

import s2s


def load_args(script: str) -> argparse.Namespace:
    if script == 'run_train_model':
        return _args_run_train()


def _args_run_train() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog='python run_train_model.py',
        description='Train sequence-to-sequence model.',
    )
    subparsers = parser.add_subparsers(dest='model_name')

    for k, v in s2s.cfg.CFG_OPTS.items():
        subparser = subparsers.add_parser(
            k,
            help=f'Train {k} sequence-to-sequence model.',
        )
        v.update_parser(subparser)

    return parser.parse_args(sys.argv[1:])