from typing import Dict, Type, Union

from s2s.dset._arith import ArithDset
from s2s.dset._base import BaseDset
from s2s.dset._news_summary import NewsSummaryDset
from s2s.dset._news_commentary import NewsTranslateDset

Dset = Union[
    ArithDset,
    NewsSummaryDset,
    NewsTranslateDset,
]

DSET_OPTS: Dict[str, Type[Dset]] = {
    ArithDset.dset_name: ArithDset,
    NewsSummaryDset.dset_name: NewsSummaryDset,
    NewsTranslateDset.dset_name: NewsTranslateDset,
}
