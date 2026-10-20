import json
import os

BASE = os.path.dirname(os.path.abspath(__file__))

# Plan A
with open(os.path.join(BASE, '..', 'results', 'multi_seed_raw.json')) as f:
    data_a = json.load(f)

print('=== Plan A (test, seed=42) ===')
seed42_a = data_a['test']['42']
print(type(seed42_a))
if isinstance(seed42_a, dict):
    for k, v in seed42_a.items():
        print(f'  {k}: {v}')
elif isinstance(seed42_a, (int, float)):
    print(f'  value: {seed42_a}')

print()
print('=== Plan A (train, seed=42) ===')
train42_a = data_a['train']['42']
if isinstance(train42_a, dict):
    for k, v in train42_a.items():
        print(f'  {k}: {v}')

print()
# Plan B
with open(os.path.join(BASE, 'results', 'multi_seed_raw.json')) as f:
    data_b = json.load(f)

print('=== Plan B top keys ===')
print(list(data_b.keys()))
if 'test' in data_b:
    print('test keys:', list(data_b['test'].keys()))
    for seed_key in data_b['test']:
        seed_data = data_b['test'][seed_key]
        print(f'  seed {seed_key}: type={type(seed_data).__name__}')
        if isinstance(seed_data, dict):
            print(f'    keys: {list(seed_data.keys())}')
            for model, metrics in seed_data.items():
                print(f'    {model}: {metrics}')
        break  # only first seed
