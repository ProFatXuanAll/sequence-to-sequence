import json
import os

from typing import Union

from s2s.path import EXP_PATH


def peek_cfg(
        exp_name: str,
        file_name: str,
        key: str,
) -> Union[bool, float, int, str]:
    file_path = os.path.join(EXP_PATH, exp_name, file_name)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f'{file_path} does not exist.')

    with open(file_path, 'r', encoding='utf-8') as cfg_file:
        return json.load(cfg_file)[key]
