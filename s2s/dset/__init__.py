from typing import Dict, Type, Union

from s2s.dset._arith import ArithDset
from s2s.dset._eng2chi import Eng2ChiDset
from s2s.dset._base import BaseDset

Dset = Union[
    ArithDset,
    Eng2ChiDset,
]

DSET_OPTS: Dict[str, Type[Dset]] = {
    ArithDset.dset_name: ArithDset,
    Eng2ChiDset.dset_name: Eng2ChiDset
}
