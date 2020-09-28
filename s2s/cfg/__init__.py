from typing import Dict, Type

from s2s.cfg._base import BaseCfg, BaseDecCfg, BaseEncCfg
from s2s.cfg._gru import GRUCfg, GRUDecCfg, GRUEncCfg
from s2s.cfg._rnn import RNNCfg, RNNDecCfg, RNNEncCfg

CFG_OPTIONS: Dict[str, Type[BaseCfg]] = {
    'gru': GRUCfg,
    'rnn': RNNCfg,
}
