import argparse

from s2s.cfg import BaseTkerCfg
from s2s.cfg.tker import TKER_CFG_OPTS
from s2s.tker import TKER_OPTS
from s2s.util import load_args, peek_cfg

if __name__ == '__main__':
    script = 'run_train_tker'
    args = load_args(script=script)

    # When perform `python run_train_tker.py continue`.
    if args.tker_name == 'continue':
        tker_name = peek_cfg(
            exp_name=args.exp_name,
            file_name=BaseTkerCfg.file_name,
            key='tker_name'
        )
        tker_cfg = TKER_CFG_OPTS[tker_name].load(exp_name=args.exp_name)
        # Add new dataset.
        for dset_name in args.dset_name:
            if dset_name not in tker_cfg.dset_name:
                tker_cfg.dset_name.append(dset_name)

        # Increase vocab size
        tker_cfg.n_vocab = max(tker_cfg.n_vocab, args.n_vocab)
    # When perform `python run_train_tker.py some_tker`.
    else:
        tker_cfg = TKER_CFG_OPTS[args.tker_name](**args.__dict__)

    tker_cfg.save(exp_name=args.exp_name)
