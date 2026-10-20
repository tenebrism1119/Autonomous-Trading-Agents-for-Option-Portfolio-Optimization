import os
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))

for model in ['a2c','ppo','sac']:
    df = pd.read_csv(os.path.join(BASE, 'results', 'run_seed42', 'weights_test', f'{model}.csv'))
    print(f'=== {model} test (first 5 rows) ===')
    print(df.head(5).to_string())
    print(f'  Total rows: {len(df)}')
    print(f'  b1 range: {df["b1"].min()} ~ {df["b1"].max()}')
    print(f'  s1 range: {df["s1"].min()} ~ {df["s1"].max()}')
    non_zero = (df[["b1","b2","b3","s1","s2","s3"]].sum(axis=1) != 0).sum()
    print(f'  Non-zero days: {non_zero}')
    print()
