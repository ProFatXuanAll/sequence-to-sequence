from typing import Dict, Type, Union

from s2s.model._gru import GRUDecModel, GRUEncModel, GRUModel
from s2s.model._lstm import LSTMDecModel, LSTMEncModel, LSTMModel
from s2s.model._rnn import RNNDecModel, RNNEncModel, RNNModel

Model = Union[
    GRUModel,
    LSTMModel,
    RNNModel,
]

MODEL_OPTS: Dict[str, Type[Model]] = {
    GRUModel.model_name: GRUModel,
    LSTMModel.model_name: LSTMModel,
    RNNModel.model_name: RNNModel,
}
