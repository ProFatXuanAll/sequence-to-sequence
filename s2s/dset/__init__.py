from typing import Dict, Type, Union

from s2s.dset._arith import ArithDset
from s2s.dset._base import BaseDset

Dset = Union[
    ArithDset,
]

DSET_OPTS: Dict[str, Type[Dset]] = {
    ArithDset.dset_name: ArithDset,
}
