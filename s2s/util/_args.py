import argparse
import sys

import s2s

def load_args(script: str) -> argparse.Namespace:
    if script == 'run_train':
        return args_run_train()

def args_run_train() -> argparse.Namespace:
    tmp_parser = argparse.ArgumentParser()
    s2s.cfg.BaseCfg.update_arg_parser(tmp_parser)
    tmp_args = tmp_parser.parse_known_args(sys.argv[1:])[0]

    parser = argparse.ArgumentParser()
    s2s.cfg.CFG_OPTIONS[tmp_args.model_name].update_arg_parser(parser)

    return parser.parse_args(sys.argv[1:])
