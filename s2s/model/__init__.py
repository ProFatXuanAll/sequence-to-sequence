from typing import Union

from s2s.model._gru import GRUDecModel, GRUEncModel, GRUModel
from s2s.model._rnn import RNNDecModel, RNNEncModel, RNNModel

Model = Union[GRUModel, RNNModel]
