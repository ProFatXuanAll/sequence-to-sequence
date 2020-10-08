import argparse
import os

import torch
import torch.utils.tensorboard

from tqdm import tqdm

from s2s.dset import DSET_OPTS
from s2s.model import MODEL_OPTS
from s2s.path import EXP_PATH
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
            '--dec_max_len',
            help='Decoder max sequence length.',
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
            '--enc_max_len',
            help='Encoder max sequence length.',
            required=True,
            type=int,
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
            '--max_norm',
            help='Gradient bound to avoid gradient explosion.',
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

    # Load encoder tokenizer and its configuration.
    enc_tknzr_cfg = load_cfg(exp_name=args.enc_tknzr_exp)
    enc_tknzr = TKNZR_OPTS[enc_tknzr_cfg['tknzr_name']].load(cfg=enc_tknzr_cfg)

    # Load decoder tokenizer and its configuration.
    dec_tknzr_cfg = load_cfg(exp_name=args.dec_tknzr_exp)
    dec_tknzr = TKNZR_OPTS[dec_tknzr_cfg['tknzr_name']].load(cfg=dec_tknzr_cfg)

    # Load training datset and create dataloader.
    dset = DSET_OPTS[args.dset_name]()
    dldr = torch.utils.data.DataLoader(
        dataset=dset,
        batch_size=args.batch_size,
        shuffle=True,
    )

    # Get model running device.
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')

    # Create model.
    model = MODEL_OPTS[args.model_name](
        dec_tknzr_cfg=dec_tknzr_cfg,
        enc_tknzr_cfg=enc_tknzr_cfg,
        model_cfg=args.__dict__,
    )
    model.train()
    model = model.to(device)

    # Create optimizer.
    optim = torch.optim.Adam(
        params=model.parameters(),
        lr=args.lr,
    )

    # Create objective function.
    objtv = torch.nn.CrossEntropyLoss()

    # Save model configuration.
    save_cfg(cfg=args.__dict__, exp_name=args.exp_name)

    # Global step.
    step = 0

    # Create experiment folder.
    exp_path = os.path.join(EXP_PATH, args.exp_name)

    if not os.path.exists(exp_path):
        os.makedirs(exp_path)

    # Create logger and log folder.
    writer = torch.utils.tensorboard.SummaryWriter(
        os.path.join(EXP_PATH, 'log', args.exp_name)
    )

    # Log average loss.
    total_loss = 0.0
    pre_total_loss = 0.0

    for cur_epoch in range(args.epoch):
        tqdm_dldr = tqdm(
            dldr,
            desc=f'epoch: {cur_epoch}, loss: {pre_total_loss:.6f}'
        )
        for batch in tqdm_dldr:
            src, src_len = enc_tknzr.batch_enc(
                batch_text=batch[0],
                max_len=args.enc_max_len,
            )
            tgt, tgt_len = dec_tknzr.batch_enc(
                batch_text=batch[1],
                max_len=args.dec_max_len,
            )

            src = torch.tensor(src).to(device)
            src_len = torch.tensor(src_len).to(device)
            tgt = torch.tensor(tgt).to(device)
            tgt_len = torch.tensor(tgt_len).to(device)

            # Forward pass.
            logits = model(
                src=src,
                src_len=src_len,
                tgt=tgt[:, :-1],
                tgt_len=tgt_len - 1,
            )

            # Calculate loss.
            loss = objtv(
                logits.reshape(-1, dec_tknzr_cfg['n_vocab']),
                tgt[:, 1:].reshape(-1),
            )

            # Accumulate loss.
            total_loss += loss.item() / args.log_step

            # Backward pass.
            loss.backward()

            # Perform gradient clipping.
            torch.nn.utils.clip_grad_norm_(
                parameters=model.parameters(),
                max_norm=args.max_norm,
            )

            # Gradient descent.
            optim.step()

            # Clean up gradient.
            optim.zero_grad()

            # Increment global step.
            step += 1

            # Save checkpoint for each `ckpt_step`.
            if step % args.ckpt_step == 0:
                torch.save(
                    model.state_dict(),
                    os.path.join(exp_path, f'model-{step}.pt'),
                )
            if step % args.log_step == 0:
                # Log average loss on CLI.
                tqdm_dldr.set_description(
                    f'epoch: {cur_epoch}, loss: {total_loss:.6f}'
                )

                # Log average loss on tensorboard.
                writer.add_scalar('loss', total_loss, step)

                # Clean up average loss.
                pre_total_loss = total_loss
                total_loss = 0.0

    # Save last checkpoint.
    torch.save(
        model.state_dict(),
        os.path.join(exp_path, f'model-{step}.pt'),
    )

    # Close logger.
    writer.close()


if __name__ == '__main__':
    main()
