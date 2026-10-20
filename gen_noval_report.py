# -*- coding: utf-8 -*-
"""
Assemble the deliverable for the NO-validation rolling backtest and its
reproduction check against the main experiment.

Reads (relative to this code folder):
  results_wf_noval/windows_metrics.csv        per-window TVM/MDD/Sharpe
  results_wf_noval/main_seed42/metrics.csv      main test-period metrics (= original split)
  results_wf_noval/pooled_metrics.csv         pooled full-period metrics
  results_wf_noval/pooled_daily_returns.csv   pooled daily returns (for the NAV chart)

Writes (deliverables):
  ../no_validation_rolling_backtest_vs_original.xlsx
  ../no_validation_rolling_backtest_NAV.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.abspath(os.path.join(HERE, '..'))
XLSX = os.path.join(OUT_DIR, 'no_validation_rolling_backtest_vs_original.xlsx')
PNG = os.path.join(OUT_DIR, 'no_validation_rolling_backtest_NAV.png')

NOVAL_DIR = os.path.join(HERE, 'results_wf_noval')
ORIG_WINDOW = 'main'
MODELS = ['a2c', 'ddpg', 'ppo', 'td3', 'sac', 'equal_weight', 'close']
DISP = {'a2c': 'A2C', 'ddpg': 'DDPG', 'ppo': 'PPO', 'td3': 'TD3', 'sac': 'SAC',
        'equal_weight': 'Equal-Weight', 'close': 'Buy&Hold(ETF)'}

# Original single-window notebook test-period (out-of-sample, 39 days) recorded values.
RECORDED_OOS = {
    'pf': {'a2c': 1.270552, 'ddpg': 4.278962, 'ppo': 2.947627,
           'td3': 0.266330, 'sac': 0.927517, 'close': 0.931863},
    'md': {'a2c': -0.203995, 'ddpg': -0.279185, 'ppo': -0.177951,
           'td3': -0.795856, 'sac': -0.419028, 'close': -0.084297},
    'sr': {'a2c': 1.826308, 'ddpg': 4.989122, 'ppo': 4.705803,
           'td3': -4.004336, 'sac': 0.231964, 'close': -3.659669},
}


def _read(path):
    return pd.read_csv(path) if os.path.exists(path) else None


def fact_sheet():
    rows = [
        ('Item', 'Content'),
        ('Original code (old experiment)', 'portfolio_rl_hs300_con.py (2026-05-09, identical to the .ipynb)'),
        ('Original data range', '2022-07-01 ~ 2023-05-31 (223 trading days, about 11 months)'),
        ('Original split', "train=date<'2023-04-01' (184 days) ; test=date>='2023-04-01' (39 days)"),
        ('Original split ratio', 'train:test = 184:39 ~ 9:2 (9 months : 2 months)'),
        ('Original validation set', 'None'),
        ('Original random seed', 'Not set'),
        ('This rolling experiment', 'Same window size as the original (9m train + 2m test, no validation set), slid over the extended data'),
        ('Rolling data range', '2022-01-01 ~ 2023-12-31 (484 trading days)'),
        ('Rolling windows / step', '7 windows, step 2 months; test periods cover 2022-10 ~ 2023-11'),
        ('Reproduction anchor window', f'{ORIG_WINDOW} = original split (train 2022-07~2023-03, test 2023-04~2023-05)'),
        ('Algorithms / steps / seed', 'a2c/ddpg/td3 50k, ppo 100k, sac 70k; seed=42; final model (no validation selection)'),
        ('Environment', 'state_space=67 (9 features); hmax=30000, initial=60000, cost 0.0001, reward_scaling=1e-1'),
    ]
    return pd.DataFrame(rows[1:], columns=list(rows[0]))


def windows_sheet():
    d = _read(os.path.join(NOVAL_DIR, 'windows_metrics.csv'))
    if d is None:
        return pd.DataFrame()
    d = d.copy()
    d['window'] = d['window'].astype(str)
    d['model_disp'] = d['model'].map(DISP)
    d['is_orig_window'] = d['window'] == ORIG_WINDOW
    piv = d.pivot(index='window', columns='model', values='tvm')
    piv = piv.reindex(columns=[m for m in MODELS if m in piv.columns])
    piv.insert(0, 'window', piv.index)
    return piv.reset_index(drop=True)


def main_repro_sheet():
    d = _read(os.path.join(NOVAL_DIR, f'{ORIG_WINDOW}_seed42', 'metrics.csv'))
    if d is None:
        return pd.DataFrame()
    d = d.set_index('model')
    rows = []
    for metric, col in [('pf', 'tvm'), ('md', 'mdd'), ('sr', 'sharpe')]:
        for name in MODELS:
            rec = RECORDED_OOS[metric].get(name)
            got = float(d.loc[name, col]) if name in d.index else np.nan
            rows.append({'metric': metric, 'model': name, 'recorded_5-9': rec,
                         'reproduced_main': got,
                         'abs_diff': (got - rec) if rec is not None else np.nan})
    return pd.DataFrame(rows)


def pooled_sheet():
    d = _read(os.path.join(NOVAL_DIR, 'pooled_metrics.csv'))
    if d is None:
        return pd.DataFrame()
    d = d.set_index('model').reset_index()
    d['model_disp'] = d['model'].map(DISP)
    return d


def make_nav():
    d = _read(os.path.join(NOVAL_DIR, 'pooled_daily_returns.csv'))
    if d is None:
        return False
    d = d.copy()
    d.index = pd.to_datetime(d.index)
    d = d[~d.index.duplicated(keep='last')]
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    fig, ax = plt.subplots(figsize=(15, 6))
    for col in d.columns:
        name = col.replace('_r', '')
        if name not in DISP:
            continue
        nav = (1 + d[col].fillna(0)).cumprod()
        style = dict(lw=2.2, color='tab:red', zorder=10) if name == 'ppo' else (
            dict(lw=1.6, color='tab:gray') if name == 'close' else dict(lw=1.0, ls='--'))
        ax.plot(nav.index, nav.values, label=DISP[name], **style)
    ax.axhline(1.0, color='black', lw=0.8, alpha=0.5)
    ax.legend()
    ax.set_ylabel('NAV (start = 1.0)')
    ax.set_title('No-validation rolling backtest - pooled out-of-sample NAV (9m train + 2m test, seed=42)')
    fig.tight_layout()
    fig.savefig(PNG, dpi=140)
    plt.close(fig)
    return True


def conclusion():
    lines = [
        '1. Window design',
        '  Uses the window size of the earliest single-window experiment: 9 months train + 2 months test',
        '  (184:39 ~ 9:2), no validation set, slid over the extended data',
        '  (2022-01-01~2023-12-31, 484 trading days) with a 2-month step, 7 windows in total.',
        f'  The split of window {ORIG_WINDOW} coincides exactly with the main experiment',
        '  (train 2022-07~2023-03, test 2023-04~2023-05), so it can serve both as the anchor for',
        '  "reproducing the earliest experiment" and, through the other 6 windows, as a way to turn',
        '  the single-window experiment into a 7-window rolling out-of-sample evaluation.',
        '',
        '2. Reproducibility',
        f'  Comparing {ORIG_WINDOW} test-period metrics with the values recorded in the original notebook:',
        '  - The Buy&Hold(ETF) benchmark pf/md/sr should match the records digit by digit (data and calendar alignment are correct);',
        '  - The numbers of the 5 DRL algorithms depend on training randomness; the original code set no seed, so',
        '    digit-by-digit agreement should not be expected. Their differences measure the random fluctuation',
        '    of the original single-window results.',
        '',
        '3. Stability (out-of-sample performance on the extended data)',
        '  1. TVM/Sharpe fluctuate greatly across windows and algorithms, so a single window is not representative;',
        '  2. The pooled full-period metrics show whether the strategy is robust overall on the extended data',
        '     (see the "pooled full-period metrics" sheet);',
        '  3. If most algorithms see NAV approach zero / blow up in some windows, the impressive single-window',
        '     result of the initial submission cannot be extrapolated. It reflects structural instability and',
        '     should only be used for diagnostics, not as evidence of robustness in the main text.',
        '',
        '4. Protocol notes',
        '  Compared with walk_forward.py (with validation: 8m train + 1m val + 2m test, checkpoint chosen by',
        '  validation Sharpe), this experiment drops the validation set, so the training period becomes 9 months.',
        '  The algorithm suite / step counts / step size / seeds are unchanged.',
    ]
    return '\n'.join(lines)


def main():
    sheets = {'1_original_experiment_facts': fact_sheet()}
    w = windows_sheet()
    if len(w):
        sheets['2_per_window_metrics_TVM'] = w
    r = main_repro_sheet()
    if len(r):
        sheets[f'3_{ORIG_WINDOW}_repro_vs_5-9_records'] = r
    p = pooled_sheet()
    if len(p):
        sheets['4_pooled_metrics'] = p
    if make_nav():
        print(f'[Save] NAV plot -> {PNG}')

    with pd.ExcelWriter(XLSX, engine='openpyxl') as xw:
        for name, df in sheets.items():
            df.to_excel(xw, sheet_name=name[:31], index=False)
        pd.DataFrame({'Conclusion': conclusion().split('\n')}).to_excel(
            xw, sheet_name='5_conclusions', index=False)
    print(f'[Save] Workbook -> {XLSX}')
    print('[Sheets]', list(sheets.keys()) + ['5_conclusions'])


if __name__ == '__main__':
    main()
