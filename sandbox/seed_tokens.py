import json
import os
import yaml

from nanochat.common import get_base_dir

seeds = set()
seed_config_file = f"{os.path.dirname(os.path.abspath(__file__))}/seed_tokens.yaml"
if os.path.exists(seed_config_file):
    with open(seed_config_file) as f:
        seed_configs = yaml.safe_load(f)
        if 'versatile_morphemes' in seed_configs:
            for seed in seed_configs["versatile_morphemes"]:
                # Can be beginning, standalone, or inside
                seeds.add(f" {seed}")
                seeds.add(f" {seed.capitalize()}")
                seeds.add(f"{seed}")
        if 'prefixes' in seed_configs:
            for seed in seed_configs["prefixes"]:
                # Always at the beginning
                seeds.add(f" {seed}")
                seeds.add(f" {seed.capitalize()}")
        if 'inner_morphemes' in seed_configs:
            for seed in seed_configs["inner_morphemes"]:
                # Always inside or at the end
                seeds.add(f"{seed}")

with open(f'{get_base_dir()}/seed_tokens.json', 'w') as f:
    seeds = list(seeds)
    seeds.sort()
    json.dump(seeds, f, indent=2)
