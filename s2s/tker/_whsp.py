import re

from typing import List
from typing import Sequence

from s2s.tker._base import BaseTker

class WhspTker(BaseTker):
    def tokenize(self, text: str) -> List[str]:
        tks = re.split(r'\s+', self.preprocess(text))
        if tks == ['']:
            return []
        return tks

    def detokenize(self, tks: Sequence[str]) -> str:
        return self.preprocess(' '.join(tks))
