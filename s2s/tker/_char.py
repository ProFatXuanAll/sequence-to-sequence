from typing import List
from typing import Sequence

from s2s.tker._base import BaseTker


class CharTker(BaseTker):

    def tokenize(self, text: str) -> List[str]:
        return list(self.preprocess(text))

    def detokenize(self, tks: Sequence[str]) -> str:
        return self.preprocess(''.join(tks))
