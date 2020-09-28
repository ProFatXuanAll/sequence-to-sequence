r"""Configuration for sequence to sequence model based on RNN."""

import argparse

from s2s.cfg._base import BaseModelCfg


class RNNCfg(BaseModelCfg):
    model_name = 'RNN'

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
            help=f'Whether to use case sensitive {cls.model_name} decoder.',
            action='store_true',
        )
        parser.add_argument(
            '--dec_n_layer',
            help=f'Number of decoder {cls.model_name} layer(s).',
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
            help=f'Whether to use case sensitive {cls.model_name} encoder.',
            action='store_true',
        )
        parser.add_argument(
            '--enc_n_layer',
            help=f'Number of encoder {cls.model_name} layer(s).',
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
            help=f'Whether to use bidirectional {cls.model_name} encoder or not.',
            action='store_true',
        )
