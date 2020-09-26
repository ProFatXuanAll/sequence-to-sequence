r"""Configuration for sequence to sequence model based on RNN."""

class RNNEncCfg:
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


class RNNDecCfg:
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


class RNNCfg:
    def __init__(
            self,
            dec_d_emb: int,
            dec_d_hid: int,
            dec_dropout: float,
            dec_is_cased: bool,
            dec_n_layer: int,
            dec_n_vocab: int,
            dec_pad_id: int,
            enc_d_emb: int,
            enc_d_hid: int,
            enc_dropout: float,
            enc_is_bidir: bool,
            enc_is_cased: bool,
            enc_n_layer: int,
            enc_n_vocab: int,
            enc_pad_id: int,
    ):
        self.dec_config = RNNDecCfg(
            d_emb=dec_d_emb,
            d_enc_hid=enc_d_hid * (enc_is_bidir + 1),
            d_hid=dec_d_hid,
            dropout=dec_dropout,
            is_cased=dec_is_cased,
            n_layer=dec_n_layer,
            n_vocab=dec_n_vocab,
            pad_id=dec_pad_id,
        )
        self.enc_config = RNNEncCfg(
            d_emb=enc_d_emb,
            d_hid=enc_d_hid,
            dropout=enc_dropout,
            is_bidir=enc_is_bidir,
            is_cased=enc_is_cased,
            n_layer=enc_n_layer,
            n_vocab=enc_n_vocab,
            pad_id=enc_pad_id,
        )
