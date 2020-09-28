from typing import Dict, Type

from s2s.cfg._base import BaseExpCfg, BaseModelCfg, BaseOptimCfg, BaseTkerCfg
from s2s.cfg._gru import GRUCfg
from s2s.cfg._rnn import RNNCfg

MODEL_CFG_OPTS: Dict[str, Type[BaseModelCfg]] = {
    'GRU': GRUCfg,
    'RNN': RNNCfg,
}
