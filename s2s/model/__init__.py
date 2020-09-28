from typing import Dict, Type, Union

from s2s.model._gru import GRUDecModel, GRUEncModel, GRUModel
from s2s.model._rnn import RNNDecModel, RNNEncModel, RNNModel

Model = Union[GRUModel, RNNModel]

MODEL_OPTS: Dict[str, Type[Model]] = {
    'GRU': GRUModel,
    'RNN': RNNModel,
}
