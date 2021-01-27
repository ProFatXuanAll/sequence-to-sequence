from typing import Dict, Type, Union

from s2s.dset._arith import ArithDset
from s2s.dset._base import BaseDset
from s2s.dset._news_summary import NewsSummaryDset
from s2s.dset._news_commentary import NewsTranslateDset_zh_en
from s2s.dset._news_commentary import NewsTranslateDset_en_zh
from s2s.dset._news_commentary import NewsTranslateDset_ru_en
from s2s.dset._news_commentary import NewsTranslateDset_en_ru
from s2s.dset._news_commentary import NewsTranslateDset_de_en
from s2s.dset._news_commentary import NewsTranslateDset_en_de
from s2s.dset._news_commentary import NewsTranslateDset_cs_en
from s2s.dset._news_commentary import NewsTranslateDset_en_cs

Dset = Union[
    ArithDset,
    NewsSummaryDset,
    NewsTranslateDset_zh_en,
    NewsTranslateDset_en_zh,
    NewsTranslateDset_ru_en,
    NewsTranslateDset_en_ru,
    NewsTranslateDset_de_en,
    NewsTranslateDset_en_de,
    NewsTranslateDset_cs_en,
    NewsTranslateDset_en_cs,
]

DSET_OPTS: Dict[str, Type[Dset]] = {
    ArithDset.dset_name: ArithDset,
    NewsSummaryDset.dset_name: NewsSummaryDset,
    NewsTranslateDset_zh_en.dset_name: NewsTranslateDset_zh_en,
    NewsTranslateDset_en_zh.dset_name: NewsTranslateDset_en_zh,
    NewsTranslateDset_ru_en.dset_name: NewsTranslateDset_ru_en,
    NewsTranslateDset_en_ru.dset_name: NewsTranslateDset_en_ru,
    NewsTranslateDset_de_en.dset_name: NewsTranslateDset_de_en,
    NewsTranslateDset_en_de.dset_name: NewsTranslateDset_en_de,
    NewsTranslateDset_cs_en.dset_name: NewsTranslateDset_cs_en,
    NewsTranslateDset_en_cs.dset_name: NewsTranslateDset_en_cs,
}
