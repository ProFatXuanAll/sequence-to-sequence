r"""Configuration for sequence to sequence model based on RNN."""

import argparse

from s2s.cfg._base import BaseCfg, BaseDecCfg, BaseEncCfg

class RNNEncCfg(BaseEncCfg):
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
    def parse_args(cls, args: argparse.Namespace):
        return cls(
            d_emb=args.enc_d_emb,
            d_hid=args.enc_d_hid,
            dropout=args.enc_dropout,
            is_bidir=args.enc_is_bidir,
            is_cased=args.enc_is_cased,
            n_layer=args.enc_n_layer,
            n_vocab=args.enc_n_vocab,
            pad_id=args.enc_pad_id,
        )

class RNNDecCfg(BaseDecCfg):
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
    def parse_args(cls, args: argparse.Namespace):
        return cls(
            d_emb=args.dec_d_emb,
            d_enc_hid=args.enc_d_hid * (args.enc_is_bidir + 1),
            d_hid=args.dec_d_hid,
            dropout=args.dec_dropout,
            is_cased=args.dec_is_cased,
            n_layer=args.dec_n_layer,
            n_vocab=args.dec_n_vocab,
            pad_id=args.dec_pad_id,
        )

class RNNCfg(BaseCfg):
    dec_cfg_cstr = RNNDecCfg
    enc_cfg_cstr = RNNEncCfg

    def __init__(
            self,
            ckpt_step: int,
            dec_cfg: RNNDecCfg,
            enc_cfg: RNNEncCfg,
            exp_name: str,
            log_step: int,
    ):
        super().__init__(
            ckpt_step=ckpt_step,
            dec_cfg=dec_cfg,
            enc_cfg=enc_cfg,
            exp_name=exp_name,
            log_step=log_step,
            model_name='rnn',
        )

    @classmethod
    def parse_args(cls, args: argparse.Namespace):
        return cls(
            ckpt_step=args.ckpt_step,
            dec_cfg=RNNDecCfg.parse_args(args=args),
            enc_cfg=RNNEncCfg.parse_args(args=args),
            exp_name=args.exp_name,
            log_step=args.log_step,
        )
