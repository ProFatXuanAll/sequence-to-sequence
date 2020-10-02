from typing import Dict, Type, Union

from s2s.cfg.tknzr import CharTknzrCfg
from s2s.cfg.tknzr import WhspTknzrCfg
from s2s.tknzr._base import BaseTknzr
from s2s.tknzr._char import CharTknzr
from s2s.tknzr._whsp import WhspTknzr

Tknzr = Union[
    CharTknzr,
    WhspTknzr,
]

TKNZR_OPTS: Dict[str, Type[Tknzr]] = {
    CharTknzrCfg.tknzr_name: CharTknzr,
    WhspTknzrCfg.tknzr_name: WhspTknzr,
}
