from typing import Dict, Type, Union

from s2s.infr._base import BaseInfr
from s2s.infr._top_1 import Top1Infr

Infr = Union[
    Top1Infr,
]

INFR_OPTS: Dict[str, Type[Infr]] = {
    Top1Infr.infr_name: Top1Infr,
}
