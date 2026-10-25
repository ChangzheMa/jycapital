"""原子状态、随机数状态和分时点选模验证。"""
import os
import random
from pathlib import Path

import torch
import numpy as np

from artifacts import write_json


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    write_json(temporary, value)
    os.replace(temporary, path)


def atomic_save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    torch.save(value, temporary)
    os.replace(temporary, path)


def rng_state(device):
    state = np.random.get_state()
    return {'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if device.type == 'cuda' else [],
            'numpy': [state[0], state[1].tolist(), *state[2:]], 'python': random.getstate()}


def restore_rng(state, device):
    torch.set_rng_state(state['torch'])
    if device.type == 'cuda':
        torch.cuda.set_rng_state_all(state['cuda'])
    value = state['numpy']
    np.random.set_state((value[0], np.asarray(value[1], dtype=np.uint32), *value[2:]))
    random.setstate(state['python'])


def cpu_state(module):
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}
