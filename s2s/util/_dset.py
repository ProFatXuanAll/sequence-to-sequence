import s2s.cfg
import s2s.dset


def load_dset(exp_cfg: s2s.cfg.BaseExpCfg) -> s2s.dset.Dset:
    return s2s.dset.DSET_OPTS[exp_cfg.dset_name]()
