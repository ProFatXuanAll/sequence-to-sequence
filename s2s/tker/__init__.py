from typing import Dict, Type, Union

from s2s.cfg.tker import CharTkerCfg
from s2s.cfg.tker import WhspTkerCfg
from s2s.tker._base import BaseTker
from s2s.tker._char import CharTker
from s2s.tker._whsp import WhspTker

Tker = Union[
    CharTker,
    WhspTker,
]

TKER_OPTS: Dict[str, Type[Tker]] = {
    CharTkerCfg.tker_name: CharTker,
    WhspTkerCfg.tker_name: WhspTker,
}
