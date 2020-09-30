from typing import Union

import s2s.cfg
import s2s.dset

def load_dset(
        cfg: Union[s2s.cfg.BaseModelCfg, s2s.cfg.BaseTkerCfg]
) -> s2s.dset.Dset:
    return s2s.dset.DSET_OPTS[cfg.dset_name]
