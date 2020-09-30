from typing import Dict, Type

from s2s.cfg._base import BaseTkerCfg
from s2s.cfg.tker._char import CharTkerCfg
from s2s.cfg.tker._whsp import WhspTkerCfg

TKER_CFG_OPTS: Dict[str, Type[BaseTkerCfg]] = {
    CharTkerCfg.tker_name: CharTkerCfg,
    WhspTkerCfg.tker_name: WhspTkerCfg,
}
