r"""Configuration for sequence to sequence model based on GRU."""

import argparse

from s2s.cfg._base import BaseCfg, BaseDecCfg, BaseEncCfg


class GRUEncCfg(BaseEncCfg):
    def __init__(
            self,
            d_emb: int,
            d_hid: int,
            dropout: float,
            is_bidir: bool,
            is_cased: bool,
            n_layer: int,
            n_vocab: int,
            pad_id: int,
    ):
        self.d_emb = d_emb
        self.d_hid = d_hid
        self.dropout = dropout
        self.is_bidir = is_bidir
        self.is_cased = is_cased
        self.n_layer = n_layer
        self.n_vocab = n_vocab
        self.pad_id = pad_id

    @classmethod
    def load_from_args(cls, args: argparse.Namespace) -> 'GRUEncCfg':
        return GRUEncCfg(
            d_emb=args.enc_d_emb,
            d_hid=args.enc_d_hid,
            dropout=args.enc_dropout,
            is_bidir=args.is_bidir,
            is_cased=args.enc_is_cased,
            n_layer=args.enc_n_layer,
            n_vocab=args.enc_n_vocab,
            pad_id=args.enc_pad_id,
        )

    @classmethod
    def update_parser(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            '--enc_d_emb',
            help='Encoder embedding dimension.',
            required=True,
            type=int,
        )
        parser.add_argument(
            '--enc_d_hid',
            help='Encoder hidden dimension.',
            required=True,
            type=int,
        )
        parser.add_argument(
            '--enc_dropout',
            help='Encoder dropout rate.',
            required=True,
            type=float,
        )
        parser.add_argument(
            '--enc_is_cased',
            help='Whether to use case sensitive GRU encoder.',
            action='store_true',
        )
        parser.add_argument(
            '--enc_n_layer',
            help='Number of encoder GRU layer(s).',
            required=True,
            type=int,
        )
        parser.add_argument(
            '--enc_n_vocab',
            help='Encoder vocabulary size.',
            required=True,
            type=int,
        )
        parser.add_argument(
            '--enc_pad_id',
            help='Encoder padding token id.',
            required=True,
            type=int,
        )
        parser.add_argument(
            '--is_bidir',
            help='Whether to use bidirectional GRU encoder or not.',
            action='store_true',
        )


class GRUDecCfg(BaseDecCfg):
    def __init__(
            self,
            d_emb: int,
            d_enc_hid: int,
            d_hid: int,
            dropout: float,
            is_cased: bool,
            n_layer: int,
            n_vocab: int,
            pad_id: int,
    ):
        self.d_emb = d_emb
        self.d_enc_hid = d_enc_hid
        self.d_hid = d_hid
        self.dropout = dropout
        self.is_cased = is_cased
        self.n_layer = n_layer
        self.n_vocab = n_vocab
        self.pad_id = pad_id

    @classmethod
    def load_from_args(cls, args: argparse.Namespace) -> 'GRUDecCfg':
        return GRUDecCfg(
            d_emb=args.dec_d_emb,
            d_enc_hid=args.enc_d_hid * (args.is_bidir + 1),
            d_hid=args.dec_d_hid,
            dropout=args.dec_dropout,
            is_cased=args.dec_is_cased,
            n_layer=args.dec_n_layer,
            n_vocab=args.dec_n_vocab,
            pad_id=args.dec_pad_id,
        )

    @classmethod
    def update_parser(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            '--dec_d_emb',
            help='Decoder embedding dimension.',
            required=True,
            type=int,
        )
        parser.add_argument(
            '--dec_d_hid',
            help='Decoder hidden dimension.',
            required=True,
            type=int,
        )
        parser.add_argument(
            '--dec_dropout',
            help='Decoder dropout rate.',
            required=True,
            type=float,
        )
        parser.add_argument(
            '--dec_is_cased',
            help='Whether to use case sensitive GRU decoder.',
            action='store_true',
        )
        parser.add_argument(
            '--dec_n_layer',
            help='Number of decoder GRU layer(s).',
            required=True,
            type=int,
        )
        parser.add_argument(
            '--dec_n_vocab',
            help='Decoder vocabulary size.',
            required=True,
            type=int,
        )
        parser.add_argument(
            '--dec_pad_id',
            help='Decoder padding token id.',
            required=True,
            type=int,
        )


class GRUCfg(BaseCfg):
    dec_cfg_cstr = GRUDecCfg
    enc_cfg_cstr = GRUEncCfg

    def __init__(
            self,
            ckpt_step: int,
            dec_cfg: GRUDecCfg,
            enc_cfg: GRUEncCfg,
            exp_name: str,
            log_step: int,
    ):
        super().__init__(
            ckpt_step=ckpt_step,
            dec_cfg=dec_cfg,
            enc_cfg=enc_cfg,
            exp_name=exp_name,
            log_step=log_step,
            model_name='GRU',
        )

    @classmethod
    def load_from_args(cls, args: argparse.Namespace) -> 'GRUCfg':
        return GRUCfg(
            ckpt_step=args.ckpt_step,
            dec_cfg=GRUDecCfg.load_from_args(args=args),
            enc_cfg=GRUEncCfg.load_from_args(args=args),
            exp_name=args.exp_name,
            log_step=args.log_step,
        )
