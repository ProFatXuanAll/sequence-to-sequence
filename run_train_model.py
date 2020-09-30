import argparse

import s2s

if __name__ == '__main__':
    script = 'run_train_model'
    args = s2s.util.load_args(script=script)
    exp_cfg = s2s.util.load_exp_cfg(args=args, script=script)
    model_cfg = s2s.util.load_model_cfg(args=args)
    model_cfg.save(exp_name=model_cfg.exp_name)

    model = s2s.util.load_model(args=args, cfg=model_cfg)
    print(model)
