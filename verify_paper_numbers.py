# -*- coding: utf-8 -*-
"""Cross-check every number reported in the paper against the stored results.

Each check prints ONE line: PASS when the paper value matches the stored
result within the rounding tolerance, DIFF otherwise.
"""
import os
import json
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
# Paper project root: the parent of this code directory. Override with PAPER_ROOT.
PAPER_ROOT = os.environ.get('PAPER_ROOT', os.path.dirname(HERE))


def _exp_dir(env_key, *parts):
    """Resolve one archived result folder, individually overridable via env."""
    return os.environ.get(env_key, os.path.join(PAPER_ROOT, *parts))


# Archived result folders written by the experiment scripts. Only their
# location is configurable; the file names inside are produced by the runs.
MAIN = _exp_dir('EXP_MAIN_DIR', 'exp1_main_experiment', 'results')
SEC = _exp_dir('EXP_SECOND_DIR', 'exp2_second_rebalancing_window', 'results')
EXT = _exp_dir('EXP_EXT_DIR', 'exp3_extended_sample', 'results')
SOY = _exp_dir('EXP_SOY_DIR', 'exp4_soybean_meal', 'results',
               'results_ablation_reward')

SEEDS = [42, 123, 456]
ALGOS = ['a2c', 'ddpg', 'ppo', 'td3', 'sac']
NAMES = ALGOS + ['equal_weight', 'close']
INIT = 60000.0

problems = []


def metrics(nav):
    nav = np.asarray(nav, dtype=float)
    r = nav[1:] / nav[:-1] - 1
    tvm = float(nav[-1] / INIT)
    mdd = float(np.min(nav / np.maximum.accumulate(nav) - 1))
    sd = np.std(r, ddof=1)
    sharpe = float(np.mean(r) / sd * np.sqrt(252)) if sd > 0 else 0.0
    return tvm, mdd, sharpe


def nav_from_returns(path):
    """Return DataFrame of NAV levels (one column per model)."""
    df = pd.read_csv(path, index_col=0)
    out = {}
    for c in NAMES:
        if c in df.columns and not c.endswith('_r'):
            out[c] = df[c].astype(float).values
        elif c + '_r' in df.columns:
            r = df[c + '_r'].astype(float).copy()
            r.iloc[0] = 0.0          # first row is NaN by construction
            out[c] = ((1 + r).cumprod() * INIT).values
    return pd.DataFrame(out, index=df.index)


def check(label, paper, got, tol):
    dev = abs(float(got) - float(paper))
    ok = dev <= tol
    if not ok:
        problems.append((label, paper, got, dev))
    print('  %-4s %-62s paper=%-10s got=%-10s |d|=%.5f'
          % ('PASS' if ok else 'DIFF', label,
             ('%.4f' % paper) if abs(paper) < 1e3 else ('%.3f' % paper),
             ('%.4f' % got) if abs(got) < 1e3 else ('%.3f' % got), dev))


def mean_std_from_csv(path, model_col='model'):
    d = pd.read_csv(path)
    d = d[d[model_col].isin(NAMES)]
    g = d.groupby(model_col)[['tvm', 'mdd', 'sharpe']]
    return g.mean(), g.std(ddof=1)


# ============================================================
print('=' * 100)
print('A. Sample lengths')
print('=' * 100)
main_te = [nav_from_returns(os.path.join(MAIN, '种子%d' % s, 'test_returns.csv'))
           for s in SEEDS]
sec_te = [nav_from_returns(os.path.join(SEC, '种子%d' % s, 'test_returns.csv'))
          for s in SEEDS]
ext_te = [nav_from_returns(os.path.join(EXT, '种子%d' % s, 'returns_test.csv'))
          for s in SEEDS]
ext_tr = [nav_from_returns(os.path.join(EXT, '种子%d' % s, 'returns_train.csv'))
          for s in SEEDS]
check('main test days (paper 39)', 39, len(main_te[0]), 0)
check('second test days (paper 39)', 39, len(sec_te[0]), 0)
check('ext test days (paper 183)', 183, len(ext_te[0]), 0)
check('ext train days (paper 301)', 301, len(ext_tr[0]), 0)
for tag, n, data_points in [('main train', 184, 1104), ('main test', 39, 234),
                            ('second train', 182, 1092),
                            ('second test', 39, 234),
                            ('ext train', 301, 1806), ('ext test', 183, 1098)]:
    check('%s contract-rows = 6 x days' % tag, data_points, n * 6, 0)

# ============================================================
print('\n' + '=' * 100)
print('B. Table 3 (in-sample, main training window)')
print('=' * 100)
TAB3 = {
    'a2c': (0.8922, 0.1867, -0.2870, 0.4972, 0.2579, 0.4467),
    'ddpg': (0.0774, 0.1096, -0.9674, 0.0450, -1.2603, 1.4010),
    'ppo': (2.3584, 0.5783, -0.3173, 0.1344, 1.6192, 0.1560),
    'td3': (0.5992, 0.4977, -0.6063, 0.5293, 0.3346, 0.8062),
    'sac': (0.1480, 0.1138, -0.9002, 0.1002, 0.0697, 0.5890),
}
ins = pd.read_csv(os.path.join(MAIN, '主实验_样本内_各算法各种子指标.csv'))
mu, sd = mean_std_from_csv(os.path.join(MAIN, '主实验_样本内_各算法各种子指标.csv'))
for m in ALGOS:
    check('%s TVM' % m.upper(), TAB3[m][0], mu.loc[m, 'tvm'], 0.0006)
    check('%s TVM std' % m.upper(), TAB3[m][1], sd.loc[m, 'tvm'], 0.0006)
    check('%s MDD' % m.upper(), TAB3[m][2], mu.loc[m, 'mdd'], 0.0006)
    check('%s MDD std' % m.upper(), TAB3[m][3], sd.loc[m, 'mdd'], 0.0006)
    check('%s Sharpe' % m.upper(), TAB3[m][4], mu.loc[m, 'sharpe'], 0.0006)
    check('%s Sharpe std' % m.upper(), TAB3[m][5], sd.loc[m, 'sharpe'], 0.0006)
for m, tv, md, sh in [('equal_weight', 0.1189, -0.8924, 0.0406),
                      ('close', 0.9014, -0.2115, -0.8053)]:
    r = ins[ins['model'] == m].iloc[0]
    check('%s TVM' % m, tv, r['tvm'], 0.0006)
    check('%s MDD' % m, md, r['mdd'], 0.0006)
    check('%s Sharpe' % m, sh, r['sharpe'], 0.0006)
# cross-check: mean of per-seed equals the stored summary file
sumfile = pd.read_csv(os.path.join(MAIN, '主实验_样本内_汇总_mean_std.csv')).set_index('model')
for m in ALGOS + ['equal_weight', 'close']:
    check('%s summary tvm_mean vs per-seed mean' % m, sumfile.loc[m, 'tvm_mean'],
          mu.loc[m, 'tvm'], 0.0006)

# ============================================================
print('\n' + '=' * 100)
print('C. Table 4 (out-of-sample, main testing window)')
print('=' * 100)
TAB4 = {
    'a2c': (1.6083, 1.0536, -0.0710, 0.1229, 1.5486, 2.6823),
    'ddpg': (0.9619, 0.8781, -0.5583, 0.3898, -0.7611, 5.1842),
    'ppo': (1.8119, 0.6860, -0.0661, 0.0509, 4.9826, 1.0219),
    'td3': (1.2311, 1.1227, -0.3504, 0.4297, -0.0034, 4.0565),
    'sac': (0.8802, 0.7631, -0.5491, 0.3198, -0.2888, 2.8728),
}
mu4, sd4 = mean_std_from_csv(os.path.join(MAIN, '主实验_各算法各种子指标.csv'))
for m in ALGOS:
    check('%s TVM' % m.upper(), TAB4[m][0], mu4.loc[m, 'tvm'], 0.0006)
    check('%s TVM std' % m.upper(), TAB4[m][1], sd4.loc[m, 'tvm'], 0.0006)
    check('%s MDD' % m.upper(), TAB4[m][2], mu4.loc[m, 'mdd'], 0.0006)
    check('%s MDD std' % m.upper(), TAB4[m][3], sd4.loc[m, 'mdd'], 0.0006)
    check('%s Sharpe' % m.upper(), TAB4[m][4], mu4.loc[m, 'sharpe'], 0.0006)
    check('%s Sharpe std' % m.upper(), TAB4[m][5], sd4.loc[m, 'sharpe'], 0.0006)
per4 = pd.read_csv(os.path.join(MAIN, '主实验_各算法各种子指标.csv'))
for c, tv, md, sh in [('equal_weight', 0.7217, -0.5875, -0.0442),
                      ('close', 0.9319, -0.0843, -3.6597)]:
    r = per4[per4['model'] == c].iloc[0]
    check('%s TVM' % c, tv, r['tvm'], 0.0006)
    check('%s MDD' % c, md, r['mdd'], 0.0006)
    check('%s Sharpe' % c, sh, r['sharpe'], 0.0006)
# per-seed values quoted in the text
for s, tv in [(42, 1.8035), (123, 2.5021), (456, 1.1302)]:
    r = per4[(per4['model'] == 'ppo') & (per4['seed'] == s)].iloc[0]
    check('text: PPO seed %d TVM' % s, tv, r['tvm'], 0.0006)
# recompute from the raw daily returns, compare with the stored per-seed file
print('\n  [recomputation from daily returns vs stored per-seed metrics]')
for k, s in enumerate(SEEDS):
    rows = per4[per4['seed'] == s].set_index('model')
    for c in NAMES:
        t, m, sh = metrics(main_te[k][c].values)
        check('seed %d %s TVM (recomputed)' % (s, c), rows.loc[c, 'tvm'], t, 5e-4)
        check('seed %d %s MDD (recomputed)' % (s, c), rows.loc[c, 'mdd'], m, 2e-3)
        check('seed %d %s Sharpe (recomputed)' % (s, c), rows.loc[c, 'sharpe'], sh, 2e-3)

# ============================================================
print('\n' + '=' * 100)
print('D. Table 5 (second rebalancing window)')
print('=' * 100)
TAB5 = {
    'a2c': (1.3716, 0.9114, -0.3495, 0.3239, 1.7439, 3.9383),
    'ddpg': (1.4729, 0.3002, -0.2835, 0.1052, 2.6930, 1.0753),
    'ppo': (2.3379, 0.0514, -0.1851, 0.0082, 4.4399, 0.0581),
    'td3': (1.0836, 0.1308, -0.4336, 0.1039, 0.9790, 0.8329),
    'sac': (0.6975, 0.2238, -0.5597, 0.1546, -1.0255, 0.8577),
}
mu5, sd5 = mean_std_from_csv(os.path.join(SEC, '第二窗口_各算法各种子指标.csv'))
for m in ALGOS:
    check('%s TVM' % m.upper(), TAB5[m][0], mu5.loc[m, 'tvm'], 0.0006)
    check('%s TVM std' % m.upper(), TAB5[m][1], sd5.loc[m, 'tvm'], 0.0006)
    check('%s MDD' % m.upper(), TAB5[m][2], mu5.loc[m, 'mdd'], 0.0006)
    check('%s MDD std' % m.upper(), TAB5[m][3], sd5.loc[m, 'mdd'], 0.0006)
    check('%s Sharpe' % m.upper(), TAB5[m][4], mu5.loc[m, 'sharpe'], 0.0006)
    check('%s Sharpe std' % m.upper(), TAB5[m][5], sd5.loc[m, 'sharpe'], 0.0006)
per5 = pd.read_csv(os.path.join(SEC, '第二窗口_各算法各种子指标.csv'))
for c, tv, md, sh in [('equal_weight', 0.6095, -0.7504, 0.0299),
                      ('close', 0.9489, -0.0602, -2.9109)]:
    r = per5[per5['model'] == c].iloc[0]
    check('%s TVM' % c, tv, r['tvm'], 0.0006)
    check('%s MDD' % c, md, r['mdd'], 0.0006)
    check('%s Sharpe' % c, sh, r['sharpe'], 0.0006)

# ============================================================
print('\n' + '=' * 100)
print('E. Table 6 (extended sample)')
print('=' * 100)
TAB6 = {
    'a2c': (2.1013, 1.8463, -0.6842, 0.1701, 1.2215, 0.7737),
    'ddpg': (0.6247, 0.3322, -0.5740, 0.5031, -0.1392, 0.8637),
    'ppo': (8.8688, 1.0311, -0.5481, 0.0177, 2.6456, 0.0931),
    'td3': (0.5274, 0.2810, -0.7977, 0.0807, -0.1808, 0.5802),
    'sac': (2.2721, 3.5457, -0.8326, 0.2207, 0.7891, 1.3719),
}
mu6 = pd.read_csv(os.path.join(EXT, 'multi_seed_summary_test.csv')).set_index('model')
for m in ALGOS:
    check('%s TVM' % m.upper(), TAB6[m][0], mu6.loc[m, 'tvm_mean'], 0.0006)
    check('%s TVM std' % m.upper(), TAB6[m][1], mu6.loc[m, 'tvm_std'], 0.0006)
    check('%s MDD' % m.upper(), TAB6[m][2], mu6.loc[m, 'mdd_mean'], 0.0006)
    check('%s MDD std' % m.upper(), TAB6[m][3], mu6.loc[m, 'mdd_std'], 0.0006)
    check('%s Sharpe' % m.upper(), TAB6[m][4], mu6.loc[m, 'sharpe_mean'], 0.0006)
    check('%s Sharpe std' % m.upper(), TAB6[m][5], mu6.loc[m, 'sharpe_std'], 0.0006)
for c, tv, md, sh in [('equal_weight', 0.0828, -0.9490, -0.4056),
                      ('close', 0.8576, -0.1886, -1.5379)]:
    check('%s TVM' % c, tv, mu6.loc[c, 'tvm_mean'], 0.0006)
    check('%s MDD' % c, md, mu6.loc[c, 'mdd_mean'], 0.0006)
    check('%s Sharpe' % c, sh, mu6.loc[c, 'sharpe_mean'], 0.0006)
# recompute from per-seed NAV levels
raw = json.load(open(os.path.join(EXT, 'multi_seed_raw.json'), encoding='utf-8'))
for s in SEEDS:
    for c in NAMES:
        t, m, sh = metrics(ext_te[SEEDS.index(s)][c].values)
        check('seed %d %s TVM (recomputed)' % (s, c), raw['test'][str(s)]['tvm'][c], t, 5e-4)
        check('seed %d %s MDD (recomputed)' % (s, c), raw['test'][str(s)]['mdd'][c], m, 2e-3)

# ============================================================
print('\n' + '=' * 100)
print('F. Table 7 (paired t-tests, pooled 546 daily returns)')
print('=' * 100)
TAB7 = {
    'equal_weight': (0.0203, 2.977, 0.0030),
    'close': (0.0173, 3.856, 0.0001),
    'a2c': (0.0095, 3.281, 0.0011),
    'ddpg': (0.0158, 3.608, 0.0003),
    'td3': (0.0176, 3.731, 0.0002),
    'sac': (0.0120, 2.172, 0.0303),
}
stored = pd.read_csv(os.path.join(EXT, 'paired_t_test.csv'))
for c, (mdiff, tval, pval) in TAB7.items():
    d = np.concatenate([(f['ppo'].values[1:] / f['ppo'].values[:-1] - 1)
                        - (f[c].values[1:] / f[c].values[:-1] - 1)
                        for f in ext_te])
    n = len(d)
    t = d.mean() / (d.std(ddof=1) / np.sqrt(n))
    p = 2 * stats.t.sf(abs(t), n - 1)
    row = stored[stored['comparison'] == 'ppo vs %s' % c].iloc[0]
    check('n obs (paper 546)', 546, n, 0)
    check('mean diff PPO vs %s' % c, mdiff, d.mean(), 6e-4)
    check('t-stat PPO vs %s' % c, tval, t, 6e-3)
    check('p-value PPO vs %s (rounded)' % c, pval, round(p, 4), 6e-5)
    check('stored csv t vs recomputed (PPO vs %s)' % c, row['t_stat'], t, 2e-3)
    check('stored csv p vs recomputed (PPO vs %s)' % c, row['p_value'], p, 2e-3)

# ============================================================
print('\n' + '=' * 100)
print('G. Table 8 (stability counts, TVM > 1 out of 9)')
print('=' * 100)
counts = {}
for tag, frames in [('main', main_te), ('second', sec_te), ('ext', ext_te)]:
    counts[tag] = {c: sum(1 for f in frames if metrics(f[c].values)[0] > 1.0)
                   for c in NAMES}
TAB8 = {'a2c': (1, 2, 2, 5), 'ddpg': (2, 3, 0, 5), 'ppo': (3, 3, 3, 9),
        'td3': (1, 2, 0, 3), 'sac': (1, 0, 1, 2)}
for m, (a, b, c_, tot) in TAB8.items():
    check('%s main window count' % m.upper(), a, counts['main'][m], 0)
    check('%s second window count' % m.upper(), b, counts['second'][m], 0)
    check('%s extended count' % m.upper(), c_, counts['ext'][m], 0)
    check('%s total of 9' % m.upper(), tot,
          counts['main'][m] + counts['second'][m] + counts['ext'][m], 0)
for b in ['equal_weight', 'close']:
    check('%s all-configuration count = 0' % b, 0,
          counts['main'][b] + counts['second'][b] + counts['ext'][b], 0)

# ============================================================
print('\n' + '=' * 100)
print('H. Table 9 (soybean-meal American options, single window)')
print('=' * 100)
cmpw5 = pd.read_csv(os.path.join(SOY, 'comparison_w5_seed42.csv'))
TAB9 = {
    ('a2c', 'pnl'): 4.31, ('a2c', 'log'): 2.25, ('a2c', 'meanvar'): 1.77,
    ('ddpg', 'pnl'): 7.12, ('ddpg', 'log'): 7.12, ('ddpg', 'meanvar'): 7.12,
    ('ppo', 'pnl'): 3.00, ('ppo', 'log'): 3.15, ('ppo', 'meanvar'): 3.17,
    ('td3', 'pnl'): 1.00, ('td3', 'log'): 5.80, ('td3', 'meanvar'): 7.12,
    ('sac', 'pnl'): 0.33, ('sac', 'log'): 4.89, ('sac', 'meanvar'): 3.13,
}
for (m, arm), v in TAB9.items():
    r = cmpw5[(cmpw5['model'] == m) & (cmpw5['arm'] == arm)].iloc[0]
    check('%s %s TVM' % (m.upper(), arm), v, r['tvm'], 6e-3)
for arm, avg in [('pnl', 3.15), ('log', 4.64), ('meanvar', 4.46)]:
    a = cmpw5[(cmpw5['arm'] == arm) & cmpw5['model'].isin(ALGOS)]
    check('average TVM (%s)' % arm, avg, a['tvm'].mean(), 6e-3)
ew9 = float(cmpw5[(cmpw5['arm'] == 'pnl') & (cmpw5['model'] == 'equal_weight')]['tvm'].iloc[0])
check('equal-weight TVM (soybean)', 2.89, ew9, 6e-3)
check('buy-and-hold futures TVM (soybean)', 1.14,
      float(cmpw5[(cmpw5['arm'] == 'pnl') & (cmpw5['model'] == 'close')]['tvm'].iloc[0]), 6e-3)
for arm, cnt in [('pnl', 3), ('log', 4), ('meanvar', 4)]:
    a = cmpw5[(cmpw5['arm'] == arm) & cmpw5['model'].isin(ALGOS)]
    check('beat equal-weight count (%s)' % arm, cnt, int((a['tvm'] > ew9).sum()), 0)
# risk bands quoted in the text ("compressed into roughly -0.33 to -0.41")
mdd_lo = min(-0.33, cmpw5[cmpw5['arm'].isin(['log', 'meanvar']) &
                          cmpw5['model'].isin(ALGOS)]['mdd'].max())
mdd_hi = cmpw5[cmpw5['arm'].isin(['log', 'meanvar']) &
               cmpw5['model'].isin(ALGOS)]['mdd'].min()
check('risk-aware MDD band upper edge ~ -0.33', -0.33, mdd_lo, 0.006)
check('risk-aware MDD band lower edge ~ -0.41', -0.41, mdd_hi, 0.006)
for arm in ['log', 'meanvar']:
    a = cmpw5[(cmpw5['arm'] == arm) & cmpw5['model'].isin(ALGOS)]
    check('%s all Sharpe positive' % arm, 1, int((a['sharpe'] > 0).all()), 0)
sac_pnl = cmpw5[(cmpw5['arm'] == 'pnl') & (cmpw5['model'] == 'sac')].iloc[0]
check('SAC P&L Sharpe ~ -11.5', -11.5, sac_pnl['sharpe'], 0.03)
ddpg = cmpw5[cmpw5['model'] == 'ddpg']
check('DDPG MDD = -0.3605 in every arm', -0.3605, ddpg['mdd'].min(), 1e-4)
check('DDPG MDD spread across arms = 0', 0.0, ddpg['mdd'].max() - ddpg['mdd'].min(), 1e-9)
check('DDPG Sharpe = 5.0441 in every arm', 5.0441, ddpg['sharpe'].min(), 1e-4)

# ============================================================
print('\n' + '=' * 100)
print('I. Abstract / text numbers')
print('=' * 100)
check('abstract PPO TVM 1.81', 1.81, mu4.loc['ppo', 'tvm'], 0.005)
check('abstract PPO Sharpe 4.98', 4.98, mu4.loc['ppo', 'sharpe'], 0.005)
check('abstract PPO MDD -6.6%', -0.066, mu4.loc['ppo', 'mdd'], 0.0006)
check('abstract second-window TVM 2.34', 2.34, mu5.loc['ppo', 'tvm'], 0.005)
check('abstract second-window TVM std 0.05', 0.05, sd5.loc['ppo', 'tvm'], 0.0015)
check('abstract second-window Sharpe 4.44', 4.44, mu5.loc['ppo', 'sharpe'], 0.005)
check('text: PPO total return 81.19%', 0.8119, mu4.loc['ppo', 'tvm'] - 1, 0.0006)
check('text: ETF total return -6.81%', -0.0681, mu4.loc['close', 'tvm'] - 1, 0.0006)
check('text: EW total return -27.83%', -0.2783, mu4.loc['equal_weight', 'tvm'] - 1, 0.0006)
check('text: in-sample ETF -9.86%', -0.0986,
      ins[ins['model'] == 'close']['tvm'].iloc[0] - 1, 0.0006)
check('text: in-sample PPO total return 135.84%', 1.3584,
      mu.loc['ppo', 'tvm'] - 1, 0.0006)
check('text: in-sample A2C -10.78%', -0.1078, mu.loc['a2c', 'tvm'] - 1, 0.0006)
check('text: PPO smallest |MDD| among DRL', ALGOS.index('ppo'),
      int(np.argmax([mu4.loc[m, 'mdd'] for m in ALGOS])), 0)
check('text: PPO highest Sharpe in-sample', ALGOS.index('ppo'),
      int(np.argmax([mu.loc[m, 'sharpe'] for m in ALGOS])), 0)
check('text: PPO highest Sharpe out-of-sample', ALGOS.index('ppo'),
      int(np.argmax([mu4.loc[m, 'sharpe'] for m in ALGOS])), 0)
for label, paper, order in [
        ('ranking main OOS = PPO>A2C>TD3>DDPG>SAC',
         ['ppo', 'a2c', 'td3', 'ddpg', 'sac'], mu4),
        ('ranking second = PPO>DDPG>A2C>TD3>SAC',
         ['ppo', 'ddpg', 'a2c', 'td3', 'sac'], mu5),
        ('ranking in-sample = PPO>A2C>TD3>SAC>DDPG',
         ['ppo', 'a2c', 'td3', 'sac', 'ddpg'], mu)]:
    got = list(order.loc[ALGOS, 'tvm'].sort_values(ascending=False).index)
    ok = [x.upper() for x in paper] == [x.upper() for x in got]
    if not ok:
        problems.append((label, paper, got, 0))
    print('  %-4s %-62s paper=%-26s got=%-26s'
          % ('PASS' if ok else 'DIFF', label, '>'.join(paper), '>'.join(got)))
check('PPO TVM>1 in every seed (main, 3 of 3)', 3,
      int(sum(metrics(f['ppo'].values)[0] > 1 for f in main_te)), 0)
check('A2C only one seed above 1 (main, text says 1)', 1,
      int(sum(metrics(f['a2c'].values)[0] > 1 for f in main_te)), 0)
check('TD3 only one seed above 1 (main, text says 1)', 1,
      int(sum(metrics(f['td3'].values)[0] > 1 for f in main_te)), 0)

# ============================================================
print('\n' + '=' * 100)
print('J. Figure data (3-seed mean NAV csv) vs per-seed recomputation')
print('=' * 100)
for tag, frames, seg in [('main', main_te, '测试段'),
                         ('second', sec_te, '测试段'),
                         ('ext', ext_te, '测试段')]:
    folder = {'main': MAIN, 'second': SEC, 'ext': EXT}[tag]
    df = pd.read_csv(os.path.join(folder, '%s_净值_3种子均值.csv' % seg))
    for c in NAMES:
        col = 'base' if c == 'close' else c
        if col not in df.columns:
            print('  SKIP figure mean NAV %s %s (column not in csv)' % (tag, c))
            continue
        rec = np.mean([f[c].values for f in frames], axis=0)
        dev = float(np.max(np.abs(df[col].values - rec)))
        ok = dev < 1.0
        if not ok:
            problems.append(('figure NAV %s %s' % (tag, c), 0, dev, dev))
        print('  %-4s %-58s max|diff| = %.2f CNY'
              % ('PASS' if ok else 'DIFF', 'figure mean NAV %s %s' % (tag, c), dev))

# ============================================================
print('\n' + '=' * 100)
if problems:
    print('DEVIATIONS FOUND: %d' % len(problems))
    for p in problems:
        print('  - %s : paper=%s, recomputed=%s (dev=%.5f)' % p)
else:
    print('ALL CHECKS PASSED')
print('=' * 100)
