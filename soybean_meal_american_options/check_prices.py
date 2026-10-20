import sys, os
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
os.chdir(BASE)
from data_loader import prepare_data

hs300_df, hv_df, expire_dates, turnover_dates, base_df = prepare_data()

print()
print('=== Option price stats (after x10 multiplier) ===')
for tic in sorted(hs300_df['tic'].unique()):
    sub = hs300_df[hs300_df['tic'] == tic]['close']
    print(f'  {tic}: min={sub.min():.1f}, mean={sub.mean():.1f}, max={sub.max():.1f}')

print()
print('=== Sample day 2023-04-03 ===')
day = hs300_df[hs300_df['date']=='2023-04-03'][['tic','close','exercise_price','contract_type']]
print(day)

print()
print('=== Price range all days ===')
print('min:', hs300_df['close'].min())
print('max:', hs300_df['close'].max())
print('mean:', hs300_df['close'].mean())
print('median:', hs300_df['close'].median())

print()
print('=== Futures (parity) ===')
print('min:', base_df['close'].min())
print('max:', base_df['close'].max())
print('mean:', base_df['close'].mean())

print()
print('=== With hmax=200, how many can 60000 buy? ===')
day_data = hs300_df[hs300_df['date']=='2023-04-03']
for _, row in day_data.iterrows():
    p = row['close']
    tic = row['tic']
    print(f'  {tic}: price={p:.1f}, max_buy={int(60000/p)}, with hmax=200 -> buy={min(int(60000/p),200)}')
