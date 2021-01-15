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
        prog='python run_infr_model.py',
        description='Infer target sequence using sequence-to-sequence model.',
    )
    parser.add_argument(
        '--ckpt',
        help='Checkpoint to evaluate.',
        required=True,
        type=int,
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
    parser.add_argument(
        '--src',
        help='Source sequence.',
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
    set_seed(model_cfg.seed)

    # Load encoder tokenizer and its configuration.
    enc_tknzr_cfg = load_cfg(exp_name=model_cfg.enc_tknzr_exp)
    enc_tknzr = TKNZR_OPTS[enc_tknzr_cfg.tknzr_name].load(cfg=enc_tknzr_cfg)

    # Load decoder tokenizer and its configuration.
    dec_tknzr_cfg = load_cfg(exp_name=model_cfg.dec_tknzr_exp)
    dec_tknzr = TKNZR_OPTS[dec_tknzr_cfg.tknzr_name].load(cfg=dec_tknzr_cfg)

    # Get model running device.
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')

    # Load model.
    model = MODEL_OPTS[model_cfg.model_name](
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

    # Output inference result.
    print(infr.gen(
        batch_text=[args.src],
        dec_max_len=model_cfg.dec_max_len,
        dec_tknzr=dec_tknzr,
        device=device,
        enc_max_len=model_cfg.enc_max_len,
        enc_tknzr=enc_tknzr,
        model=model,
    )[0])

if __name__ == '__main__':
    main()
