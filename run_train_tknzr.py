import argparse

from s2s.dset import DSET_OPTS
from s2s.tknzr import TKNZR_OPTS
from s2s.util import save_cfg


def parse_arg() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog='python run_train_tknzr.py',
        description='Train tokenizer.',
    )

    dset_choices = []
    for dset_name in DSET_OPTS:
        dset_choices.extend([
            f'{dset_name}.src',
            f'{dset_name}.tgt',
        ])

    parser.add_argument(
        'tknzr_name',
        choices=TKNZR_OPTS.keys(),
        help='Tokenizer name',
        type=str,
    )
    parser.add_argument(
        '--dset_name',
        choices=dset_choices,
        help='Name(s) of the dataset(s) to train tokenizer.',
        nargs='+',
        required=True,
    )
    parser.add_argument(
        '--exp_name',
        help='Current experiment name.',
        required=True,
        type=str,
    )
    parser.add_argument(
        '--is_cased',
        action='store_true',
        help='Whether to convert all upper cases letter into lower cases.',
    )
    parser.add_argument(
        '--min_count',
        help='Minimum frequency for tokens to be in vocabulary.',
        required=True,
        type=int,
    )
    parser.add_argument(
        '--n_vocab',
        help='Tokenizer vocabulary size.',
        required=True,
        type=int,
    )

    return parser.parse_args()


def main():
    r"""Main function."""
    # Load command line arguments.
    args = parse_arg()

    # Sort dataset list for readability.
    args.dset_name.sort()

    # Create tokenizer.
    tknzr = TKNZR_OPTS[args.tknzr_name](cfg=args)

    # Build tokenizer vocabulary.
    for dset_name in args.dset_name:
        dset = DSET_OPTS[dset_name[:-4]]()
        if '.src' in dset_name:
            tknzr.build_vocab(dset.all_src())
        if '.tgt' in dset_name:
            tknzr.build_vocab(dset.all_tgt())

    # Save tokenizer and its configuration.
    save_cfg(cfg=args.__dict__, exp_name=args.exp_name)
    tknzr.save(exp_name=args.exp_name)


if __name__ == '__main__':
    main()
