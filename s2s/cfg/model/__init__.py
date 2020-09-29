from typing import Dict, Type

from s2s.cfg._base import BaseModelCfg
from s2s.cfg.model._gru import GRUModelCfg
from s2s.cfg.model._rnn import RNNModelCfg

MODEL_CFG_OPTS: Dict[str, Type[BaseModelCfg]] = {
    GRUModelCfg.model_name: GRUModelCfg,
    RNNModelCfg.model_name: RNNModelCfg,
}
