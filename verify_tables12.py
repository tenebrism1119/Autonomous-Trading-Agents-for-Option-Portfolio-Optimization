# -*- coding: utf-8 -*-
"""Verify Tables 1-2 (descriptive statistics of the six selected contracts)."""
import os
# Default to the repository-local data folder; override with DATA_ROOT.
os.environ.setdefault('DATA_ROOT',
                      os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   'data', 'data'))
import numpy as np
import pandas as pd
from data_loader import prepare_data

hs300_df, hv_df, expire_dates, turnover_dates, base_df = prepare_data()

d = hs300_df.merge(hv_df, on='date')
cols = ['close', 'volume', 'money', 'exercise_price', 'base_p', 'diff_days', 'hv']

TAB1 = {  # July 2022 -- March 2023
    'close': (0.0973, 0.0900, 0.0001, 0.0258, 0.0726, 0.1470, 0.4574),
    'volume': (106048.0688, 78443.2165, 1094.0, 38958.0, 94388.5, 156029.25, 417464.0),
    'money': (7142996.0, 5882939.0, 110.0, 2216037.0, 6037087.0, 10452639.0, 36031468.0),
    'exercise_price': (4.0506, 0.2085, 3.60, 3.90, 4.10, 4.20, 4.40),
    'base_p': (4.0462, 0.1814, 3.5650, 3.9192, 4.0435, 4.1542, 4.5210),
    'diff_days': (15.8859, 9.3014, 1.0, 8.0, 15.0, 23.0, 35.0),
    'hv': (0.0100, 0.0027, 0.0059, 0.0080, 0.0098, 0.0105, 0.0175),
}
TAB2 = {  # April -- May 2023
    'close': (0.0759, 0.0677, 0.0001, 0.0169, 0.0560, 0.1181, 0.2600),
    'volume': (96752.9701, 62275.3321, 4259.0, 42871.0, 90768.5, 145721.75, 238258.0),
    'money': (5187196.0, 3787430.0, 3326.0, 1589311.0, 5184171.0, 8118809.0, 16134069.0),
    'exercise_price': (3.9654, 0.0947, 3.80, 3.90, 4.00, 4.00, 4.10),
    'base_p': (3.9996, 0.0934, 3.8020, 3.9500, 3.9950, 4.0880, 4.1520),
    'diff_days': (15.0000, 9.5827, 1.0, 7.0, 14.0, 21.0, 35.0),
    'hv': (0.0078, 0.0010, 0.0056, 0.0072, 0.0080, 0.0084, 0.0095),
}

def stat(v):
    return (v.mean(), v.std(ddof=1), v.min(), v.quantile(.25),
            v.median(), v.quantile(.75), v.max())

fails = 0
for label, s, e, ref in [
        ('Table 1 (train 2022-07-01..2023-03-31)', '2022-07-01', '2023-03-31', TAB1),
        ('Table 2 (test 2023-04-03..2023-05-31)', '2023-04-03', '2023-05-31', TAB2)]:
    seg = d[(d['date'] >= s) & (d['date'] <= e)]
    print('\n=== %s  (%d rows, %d days) ===' % (label, len(seg), seg['date'].nunique()))
    for c in cols:
        got = stat(seg[c].astype(float))
        ref_v = ref[c]
        for gotv, refv, nm in zip(got, ref_v, ['mean', 'std', 'min', 'q25', 'q50', 'q75', 'max']):
            dev = abs(gotv - refv)
            tol = 0.00015 if c in ('close', 'hv', 'exercise_price', 'base_p') else max(0.05, abs(refv) * 2e-6)
            if dev > tol:
                fails += 1
                print('  DIFF %-15s %-4s paper=%-14s got=%-14s |d|=%.6f'
                      % (c, nm, refv, round(gotv, 6), dev))
print('\nTOTAL DIFFS:', fails)

# ETF decline claims
print('\n=== ETF price path ===')
for a, b, note in [('2022-01-04', '2023-12-29', 'Jan 2022 -> Dec 2023 (text: ~30%)'),
                   ('2022-07-01', '2023-03-31', 'train window (text: -9.86%)'),
                   ('2023-04-03', '2023-05-31', 'test window (text: -6.81%)')]:
    x = base_df[(base_df['date'] >= a) & (base_df['date'] <= b)]['close']
    print('  %-42s %8.4f -> %8.4f  change=%+.2f%%'
          % (note, x.iloc[0], x.iloc[-1], (x.iloc[-1] / x.iloc[0] - 1) * 100))
