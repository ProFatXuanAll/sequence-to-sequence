import re

from typing import List
from typing import Sequence

from s2s.tknzr._base import BaseTknzr


class WhspTknzr(BaseTknzr):
    tknzr_name = 'whitespace'

    def tknz(self, text: str) -> List[str]:
        tks = re.split(r'\s+', self.preprocess(text))
        if tks == ['']:
            return []
        return tks

    def dtknz(self, tks: Sequence[str]) -> str:
        return self.preprocess(' '.join(tks))
