import argparse

import s2s

if __name__ == '__main__':
    args = s2s.util.load_args('run_train_model')

    cfg = s2s.util.load_cfg(args=args)
    cfg.save()

    model = s2s.util.load_model(args=args, cfg=cfg)
    print(model)
