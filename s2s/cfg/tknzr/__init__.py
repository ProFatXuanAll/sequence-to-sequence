from typing import Dict, Type

from s2s.cfg._base import BaseTknzrCfg
from s2s.cfg.tknzr._char import CharTknzrCfg
from s2s.cfg.tknzr._whsp import WhspTknzrCfg

TKNZR_CFG_OPTS: Dict[str, Type[BaseTknzrCfg]] = {
    CharTknzrCfg.tknzr_name: CharTknzrCfg,
    WhspTknzrCfg.tknzr_name: WhspTknzrCfg,
}
