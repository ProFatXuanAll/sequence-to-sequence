from typing import Dict, Type, Union

from s2s.tknzr._base import BaseTknzr
from s2s.tknzr._char import CharTknzr
from s2s.tknzr._whsp import WhspTknzr

Tknzr = Union[
    CharTknzr,
    WhspTknzr,
]

TKNZR_OPTS: Dict[str, Type[Tknzr]] = {
    CharTknzr.tknzr_name: CharTknzr,
    WhspTknzr.tknzr_name: WhspTknzr,
}
