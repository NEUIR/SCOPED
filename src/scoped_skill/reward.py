"""Use the same capability reward entrypoint as the expert's RL configuration."""
from functools import lru_cache
import importlib.util
import math
import yaml

from .common import ROOT, digest
from .judge import extract_answer


def reward_identity(capability):
    config_path = ROOT / 'configs/rl' / (capability + '.yaml')
    config = yaml.safe_load(config_path.read_text())
    entry = config['worker']['reward']['reward_function']
    filename, function = entry.rsplit(':', 1)
    path = ROOT / filename
    return {'entrypoint': entry, 'source_sha256': digest(path), 'rl_config_sha256': digest(config_path)}


@lru_cache(maxsize=4)
def load_reward(capability):
    identity = reward_identity(capability)
    filename, function = identity['entrypoint'].rsplit(':', 1)
    spec = importlib.util.spec_from_file_location('scoped_reward_' + capability, ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, function), getattr(module, 'extract_answer', extract_answer)


def score_trajectory(capability, trajectory, ground_truth):
    compute, extract = load_reward(capability)
    result = compute([{'response': trajectory, 'ground_truth': ground_truth}])
    if len(result) != 1:
        raise ValueError('Capability reward returned an invalid batch')
    score = float(result[0]['overall'])
    if not math.isfinite(score):
        raise ValueError('Capability reward returned a non-finite score')
    return score, extract(trajectory)
