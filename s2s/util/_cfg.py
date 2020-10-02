import argparse
import json
import os

from typing import Dict

from s2s.path import EXP_PATH


def save_cfg(cfg: Dict, exp_name: str, file_name: str) -> None:
    exp_path = os.path.join(EXP_PATH, exp_name)
    file_path = os.path.join(exp_path, file_name)

    if not os.path.exists(exp_path):
        os.makedirs(exp_path)

    with open(file_path, 'w', encoding='utf-8') as output:
        json.dump(cfg, output, ensure_ascii=False, indent=2)


def load_cfg(cfg):
    pass
