from typing import Dict, Type

from s2s.cfg._base import BaseCfg, BaseDecCfg, BaseEncCfg
from s2s.cfg._gru import GRUCfg, GRUDecCfg, GRUEncCfg
from s2s.cfg._rnn import RNNCfg, RNNDecCfg, RNNEncCfg

CFG_OPTS: Dict[str, Type[BaseCfg]] = {
    'GRU': GRUCfg,
    'RNN': RNNCfg,
}
