import argparse
import os

import torch
import torch.utils.tensorboard

from tqdm import tqdm

from s2s.dset import DSET_OPTS
from s2s.infr import INFR_OPTS
from s2s.model import MODEL_OPTS
from s2s.path import EXP_PATH
from s2s.tknzr import TKNZR_OPTS
from s2s.util import load_cfg, load_model_from_ckpt, set_seed


def parse_arg() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog='python run_eval_model.py',
        description='Evaluate sequence-to-sequence model.',
    )

    parser.add_argument(
        '--batch_size',
        help='Evaluation batch size.',
        required=True,
        type=int,
    )
    parser.add_argument(
        '--ckpt',
        help='Checkpoint to evaluate.',
        required=True,
        type=int,
    )
    parser.add_argument(
        '--dset_name',
        choices=DSET_OPTS.keys(),
        help='Name of the dataset to evaluate model.',
        required=True,
        type=str,
    )
    parser.add_argument(
        '--exp_name',
        help='Current experiment name.',
        required=True,
        type=str,
    )
    parser.add_argument(
        '--infr_name',
        choices=INFR_OPTS.keys(),
        help='Inference method.',
        required=True,
        type=str,
    )

    return parser.parse_args()


@torch.no_grad()
def main():
    r"""Main function."""
    # Load command line arguments.
    args = parse_arg()

    # Load model configuration.
    model_cfg = load_cfg(exp_name=args.exp_name)

    # Control random seed.
    set_seed(model_cfg['seed'])

    # Load encoder tokenizer and its configuration.
    enc_tknzr_cfg = load_cfg(exp_name=model_cfg['enc_tknzr_exp'])
    enc_tknzr = TKNZR_OPTS[enc_tknzr_cfg['tknzr_name']].load(cfg=enc_tknzr_cfg)

    # Load decoder tokenizer and its configuration.
    dec_tknzr_cfg = load_cfg(exp_name=model_cfg['dec_tknzr_exp'])
    dec_tknzr = TKNZR_OPTS[dec_tknzr_cfg['tknzr_name']].load(cfg=dec_tknzr_cfg)

    # Load evaluation datset and create dataloader.
    dset = DSET_OPTS[args.dset_name]()
    dldr = torch.utils.data.DataLoader(
        dataset=dset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    # Get model running device.
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')

    # Load model.
    model = MODEL_OPTS[model_cfg['model_name']](
        dec_tknzr_cfg=dec_tknzr_cfg,
        enc_tknzr_cfg=enc_tknzr_cfg,
        model_cfg=model_cfg,
    )
    model = load_model_from_ckpt(
        ckpt=args.ckpt,
        exp_name=args.exp_name,
        model=model,
    )
    model.eval()
    model = model.to(device)

    # Load inference method.
    infr = INFR_OPTS[args.infr_name](**args.__dict__)

    # Record batch inference result.
    all_pred = []
    for batch in tqdm(dldr):
        all_pred.extend(infr.gen(
            batch_text=batch[0],
            dec_max_len=model_cfg['dec_max_len'],
            dec_tknzr=dec_tknzr,
            device=device,
            enc_max_len=model_cfg['enc_max_len'],
            enc_tknzr=enc_tknzr,
            model=model,
        ))

    # Output all dataset result.
    print(DSET_OPTS[args.dset_name].batch_eval(
        batch_tgt=dset.all_tgt(),
        batch_pred=all_pred,
    ))
    print(DSET_OPTS[args.dset_name].bleu_score(
        batch_tgt=dset.all_tgt(),
        batch_pred=all_pred,
    ))


if __name__ == '__main__':
    main()
