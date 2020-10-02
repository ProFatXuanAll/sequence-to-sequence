from typing import List
from typing import Sequence

from s2s.tknzr._base import BaseTknzr


class CharTknzr(BaseTknzr):
    tknzr_name = 'character'

    def tknz(self, text: str) -> List[str]:
        return list(self.preprocess(text))

    def dtknz(self, tks: Sequence[str]) -> str:
        return self.preprocess(''.join(tks))
