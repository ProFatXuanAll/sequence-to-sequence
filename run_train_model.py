import argparse

import torch

from s2s.dset import DSET_OPTS
from s2s.model import MODEL_OPTS
from s2s.tknzr import TKNZR_OPTS
from s2s.util import load_cfg, save_cfg, set_seed

def parse_arg() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog='python run_train_model.py',
        description='Train sequence-to-sequence model.',
    )

    subparsers = parser.add_subparsers(dest='model_name')

    for model_name, model_cstr in MODEL_OPTS.items():
        subparser = subparsers.add_parser(
            model_name,
            help=f'Train {model_name} sequence-to-sequence model.',
        )
        subparser.add_argument(
            '--batch_size',
            help='Training batch size.',
            required=True,
            type=int,
        )
        subparser.add_argument(
            '--ckpt_step',
            help='Checkpoint save interval.',
            required=True,
            type=int,
        )
        subparser.add_argument(
            '--dec_tknzr_exp',
            help='Experiment name of the decoder paired tokenizer.',
            required=True,
            type=str,
        )
        subparser.add_argument(
            '--dset_name',
            choices=DSET_OPTS.keys(),
            help='Name of the dataset to train model.',
            required=True,
            type=str,
        )
        subparser.add_argument(
            '--enc_tknzr_exp',
            help='Experiment name of the encoder paired tokenizer.',
            required=True,
            type=str,
        )
        subparser.add_argument(
            '--epoch',
            help='Number of training epochs.',
            required=True,
            type=int,
        )
        subparser.add_argument(
            '--exp_name',
            help='Current experiment name.',
            required=True,
            type=str,
        )
        subparser.add_argument(
            '--log_step',
            help='Performance log interval.',
            required=True,
            type=int,
        )
        subparser.add_argument(
            '--lr',
            help='Gradient decent learning rate.',
            required=True,
            type=float,
        )
        subparser.add_argument(
            '--seed',
            help='Control random seed.',
            required=True,
            type=int,
        )
        model_cstr.update_subparser(subparser=subparser)

    return parser.parse_args()


def main():
    r"""Main function."""
    # Load command line arguments.
    args = parse_arg()

    # Control random seed.
    set_seed(args.seed)

    # Load encoder tokenizer.
    enc_tknzr_cfg = load_cfg(exp_name=args.enc_tknzr_exp)
    enc_tknzr = TKNZR_OPTS[enc_tknzr_cfg['tknzr_name']].load(cfg=enc_tknzr_cfg)

    # Load decoder tokenizer.
    dec_tknzr_cfg = load_cfg(exp_name=args.dec_tknzr_exp)
    dec_tknzr = TKNZR_OPTS[dec_tknzr_cfg['tknzr_name']].load(cfg=dec_tknzr_cfg)

    # Load datset.
    dset = DSET_OPTS[args.dset_name]()

    # Create new model.
    model = MODEL_OPTS[args.model_name](
        dec_tknzr_cfg=dec_tknzr_cfg,
        enc_tknzr_cfg=enc_tknzr_cfg,
        model_cfg=args.__dict__,
    )

    # Save model configuration.
    save_cfg(cfg=args.__dict__, exp_name=args.exp_name)


if __name__ == '__main__':
    main()