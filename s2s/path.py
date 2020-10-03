
r"""Path variables.

Usages:
    import s2s.path

    data_path = s2s.path.DATA_PATH
    exp_path = s2s.path.EXP_PATH
"""

# built-in modules

import os

PROJECT_ROOT = os.path.abspath(os.path.join(
    os.path.abspath(__file__),
    os.pardir,
    os.pardir
))
DATA_PATH = os.path.join(PROJECT_ROOT, 'data')
EXP_PATH = os.path.join(PROJECT_ROOT, 'exp')
CFG_NAME = 'cfg.json'
