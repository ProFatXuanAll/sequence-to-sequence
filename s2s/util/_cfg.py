import argparse
import json
import os

from typing import Dict

from s2s.path import CFG_NAME, EXP_PATH


def save_cfg(cfg: Dict, exp_name: str) -> None:
    exp_path = os.path.join(EXP_PATH, exp_name)
    file_path = os.path.join(exp_path, CFG_NAME)

    if not os.path.exists(exp_path):
        os.makedirs(exp_path)

    with open(file_path, 'w', encoding='utf-8') as output_file:
        json.dump(
            cfg,
            output_file,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )


def load_cfg(exp_name: str) -> Dict:
    file_path = os.path.join(EXP_PATH, exp_name, CFG_NAME)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f'{file_path} does not exist.')

    with open(file_path, 'r', encoding='utf-8') as input_file:
        return json.load(input_file)
