import json
import os
import yaml

from nanochat.common import get_base_dir

seeds = set()
with open(f'{os.path.dirname(os.path.abspath(__file__))}/seed_tokens.yaml', 'r') as f:
    seeds.update(yaml.safe_load(f)['seeds'])

with open(f'{get_base_dir()}/seed_tokens.json', 'w') as f:
    seeds = list(seeds)
    seeds.sort()
    json.dump(seeds, f, indent=2)
