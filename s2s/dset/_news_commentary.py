from s2s.dset._base_news_commentary import BaseNewsTranslateDset


class NewsTranslateDset_zh_en(BaseNewsTranslateDset):
    dset_name = 'news_translate_zh-en'

    def __init__(self):
        super().__init__('zh', 'en')


class NewsTranslateDset_en_zh(BaseNewsTranslateDset):
    dset_name = 'news_translate_en-zh'

    def __init__(self):
        super().__init__('en', 'zh')


class NewsTranslateDset_ru_en(BaseNewsTranslateDset):
    dset_name = 'news_translate_ru-en'

    def __init__(self):
        super().__init__('ru', 'en')


class NewsTranslateDset_en_ru(BaseNewsTranslateDset):
    dset_name = 'news_translate_en-ru'

    def __init__(self):
        super().__init__('en', 'ru')


class NewsTranslateDset_de_en(BaseNewsTranslateDset):
    dset_name = 'news_translate_de-en'

    def __init__(self):
        super().__init__('de', 'en')


class NewsTranslateDset_en_de(BaseNewsTranslateDset):
    dset_name = 'news_translate_en-de'

    def __init__(self):
        super().__init__('en', 'de')


class NewsTranslateDset_cs_en(BaseNewsTranslateDset):
    dset_name = 'news_translate_cs-en'

    def __init__(self):
        super().__init__('cs', 'en')


class NewsTranslateDset_en_cs(BaseNewsTranslateDset):
    dset_name = 'news_translate_en-cs'

    def __init__(self):
        super().__init__('en', 'cs')
