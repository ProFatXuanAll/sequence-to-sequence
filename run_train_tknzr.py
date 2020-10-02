import argparse

from typing import Tuple

from s2s.cfg import BaseTknzrCfg
from s2s.cfg.tknzr import TKNZR_CFG_OPTS
from s2s.dset import DSET_OPTS
from s2s.tknzr import BaseTknzr, TKNZR_OPTS
from s2s.util import load_args, peek_cfg


def main():
    r"""Main function."""
    # Load command line arguments.
    args = load_args(script='run_train_tknzr')

    # Perform `python run_train_tknzr.py continue`.
    if args.tknzr_name == 'continue':
        tknzr_cfg, tknzr = get_old_tknzr_and_cfg(args=args)
    # Perform `python run_train_tknzr.py some_tknzr`.
    else:
        tknzr_cfg, tknzr = get_new_tknzr_and_cfg(args=args)

    # Build tokenizer vocabulary. Same dataset can be used to build vocabulary
    # again when `n_vocab` increase, so no need to filter out used dataset.
    for dset_name in args.dset_name:
        dset = DSET_OPTS[dset_name[:-4]]()
        if '.src' in dset_name:
            tknzr.build_vocab(dset.all_src())
        if '.tgt' in dset_name:
            tknzr.build_vocab(dset.all_tgt())

    # Save tokenizer and its configuration.
    tknzr_cfg.save(exp_name=args.exp_name)
    tknzr.save(exp_name=args.exp_name)


def get_new_tknzr_and_cfg(
        args: argparse.Namespace,
) -> Tuple[BaseTknzrCfg, BaseTknzr]:
    # Create new tokenizer configuration.
    tknzr_cfg = TKNZR_CFG_OPTS[args.tknzr_name](**args.__dict__)

    # Sort dataset list for readability.
    tknzr_cfg.dset_name.sort()

    # Create new tokenizer.
    tknzr = TKNZR_OPTS[args.tknzr_name](tknzr_cfg=tknzr_cfg)

    return tknzr, tknzr_cfg


def get_old_tknzr_and_cfg(
        args: argparse.Namespace,
) -> Tuple[BaseTknzrCfg, BaseTknzr]:
    # Get tokenizer name from configuration file.
    tknzr_name = peek_cfg(
        exp_name=args.exp_name,
        file_name=BaseTknzrCfg.file_name,
        key='tknzr_name',
    )

    # Polymorphically load tokenizer configuration.
    tknzr_cfg = TKNZR_CFG_OPTS[tknzr_name].load(exp_name=args.exp_name)

    # Add new dataset. Same dataset must only appear once.
    for dset_name in args.dset_name:
        if dset_name not in tknzr_cfg.dset_name:
            tknzr_cfg.dset_name.append(dset_name)

    # Sort dataset list for readability.
    tknzr_cfg.dset_name.sort()

    # Increase vocab size.
    tknzr_cfg.n_vocab = max(tknzr_cfg.n_vocab, args.n_vocab)

    # Polymorphically load tokenizer.
    tknzr = TKNZR_OPTS[tknzr_name].load(tknzr_cfg=tknzr_cfg)

    return tknzr, tknzr_cfg


if __name__ == '__main__':
    main()
