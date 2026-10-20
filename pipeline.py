# -*- coding: utf-8 -*-
"""Shared utilities for the experiments reported in the paper.

This module replaces the helpers that previously lived in the abandoned
validation-based pipeline. It contains only the validation-free building
blocks:

* window definitions for the rolling backtest
* full-sample data loading and feature engineering
* period slicing, mean/std summaries and paired t-tests

Window naming follows the manuscript. The rolling backtest enumerates seven
consecutive windows; two of them correspond to the configurations reported in
the paper:

===============  ==========================================================
Tag              Manuscript section
===============  ==========================================================
``main``         Section 4.2, Main experiment
                 (train 2022-07-01 -- 2023-03-31, test 2023-04-03 -- 2023-05-31)
``second``       Section 4.3, Second rebalancing window
                 (train 2023-01-03 -- 2023-09-28, test 2023-10-09 -- 2023-11-30)
``rolling{1..7}``  Other windows of the rolling backtest (not reported)
===============  ==========================================================
"""

import os

import numpy as np
import pandas as pd
from finrl.meta.preprocessor.preprocessors import FeatureEngineer
from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3

from config import PPO_PARAMS, TD3_PARAMS, SAC_PARAMS
from data_loader import prepare_data, calculate_turbulence, calculate_bs_greeks


# ----------------------------------------------------------------------------
# Experiment constants
# ----------------------------------------------------------------------------
SEED = 42
ALGOS = ['a2c', 'ddpg', 'ppo', 'td3', 'sac']
ALL_NAMES = ALGOS + ['equal_weight', 'close']
DISPLAY = {'a2c': 'A2C', 'ddpg': 'DDPG', 'ppo': 'PPO', 'td3': 'TD3',
           'sac': 'SAC', 'equal_weight': 'Equal-Weight', 'close': 'Buy&Hold(ETF)'}

# PPO uses 200,000 steps, matching the main experiment in the paper.
WF_TIMESTEPS = {'a2c': 50000, 'ddpg': 50000, 'ppo': 200000, 'td3': 50000, 'sac': 70000}

MODEL_CLS = {'a2c': A2C, 'ddpg': DDPG, 'ppo': PPO, 'td3': TD3, 'sac': SAC}
MODEL_KWARGS = {'a2c': None, 'ddpg': None, 'ppo': PPO_PARAMS,
                'td3': TD3_PARAMS, 'sac': SAC_PARAMS}

N_WINDOWS = 7

# Indices (0-based) of the two windows reported in the paper.
MAIN_WINDOW_INDEX = 3
SECOND_WINDOW_INDEX = 6


def window_tag(i):
    """Return the manuscript-aligned tag of rolling window ``i`` (0-based)."""
    if i == MAIN_WINDOW_INDEX:
        return 'main'
    if i == SECOND_WINDOW_INDEX:
        return 'second'
    return 'rolling%d' % (i + 1)


def build_windows():
    """Build the date boundaries of the seven rolling windows.

    Window i starts at 2022-01 + 2*i months:
      train = [start, start+9 months)
      test  = [start+9 months, start+11 months)
    """
    wins = []
    for i in range(N_WINDOWS):
        base_m = 1 + 2 * i

        def month_start(k):
            mm = base_m + k
            yy = 2022 + (mm - 1) // 12
            mm = (mm - 1) % 12 + 1
            return '%04d-%02d-01' % (yy, mm)

        wins.append({
            'idx': i,
            'tag': window_tag(i),
            'train_s': month_start(0),
            'test_s': month_start(9),
            'test_e_excl': month_start(11),
        })
    return wins


WF_WINDOWS = build_windows()


# ----------------------------------------------------------------------------
# Data helpers
# ----------------------------------------------------------------------------
def check_and_make_directories(directories):
    for d in directories:
        if not os.path.exists('./' + d):
            os.makedirs('./' + d)


def step1_data():
    """Load option data and select the six contracts of each rebalancing period."""
    print('\n' + '=' * 60)
    print('Step 1: Data Loading & Contract Selection')
    print('=' * 60)
    hs300_df, hv_df, expire_dates, turnover_dates, base_df = prepare_data()
    return hs300_df, hv_df, expire_dates, turnover_dates, base_df


def load_full_data():
    """Load the full sample and run feature engineering, without any date split."""
    print('\n' + '=' * 60)
    print('Step A: Data Loading & Feature Engineering (full period)')
    print('=' * 60)
    hs300_df, hv_df, expire_dates, turnover_dates, base_df = prepare_data()

    hs300_df = calculate_bs_greeks(hs300_df)
    print('  BS Greeks: iv/delta/gamma/vega/moneyness added')

    fe = FeatureEngineer(
        use_technical_indicator=False, use_turbulence=False,
        user_defined_feature=False,
    )
    processed_df = fe.preprocess_data(hs300_df)

    turb = calculate_turbulence(hs300_df)
    turb['turbulence'] = turb.apply(
        lambda x: 100 if x.date in expire_dates else 0, axis=1)
    processed_df = processed_df.merge(turb, on='date')
    processed_df = processed_df.merge(hv_df, on='date')
    processed_df.index = [int(i / 6) for i in list(processed_df.index)]

    print('  Full data: %d rows, %d days, %s ~ %s' % (
        len(processed_df), processed_df['date'].nunique(),
        processed_df['date'].min(), processed_df['date'].max()))
    return processed_df, expire_dates, turnover_dates, base_df


def slice_period(processed_df, s, e_excl):
    """Slice data to [s, e_excl) and rebuild the integer index required by the env."""
    d = processed_df.loc[(processed_df.date >= s) & (processed_df.date < e_excl)].copy()
    d.index = d['date'].factorize()[0]
    return d


# ----------------------------------------------------------------------------
# Reporting helpers
# ----------------------------------------------------------------------------
def _period_summary(metrics_dict, seeds, model_names):
    """Build a mean +/- std summary DataFrame for one period."""
    rows = []
    for model in model_names:
        tvms = [float(metrics_dict[s].loc[model, 'tvm']) for s in seeds]
        mdds = [float(metrics_dict[s].loc[model, 'mdd']) for s in seeds]
        srs = [float(metrics_dict[s].loc[model, 'sharpe']) for s in seeds]
        rows.append({
            'model': model,
            'tvm_mean': np.mean(tvms),
            'tvm_std': np.std(tvms, ddof=1) if len(tvms) > 1 else 0.0,
            'mdd_mean': np.mean(mdds),
            'mdd_std': np.std(mdds, ddof=1) if len(mdds) > 1 else 0.0,
            'sharpe_mean': np.mean(srs),
            'sharpe_std': np.std(srs, ddof=1) if len(srs) > 1 else 0.0,
        })
    return pd.DataFrame(rows).set_index('model')


def _print_summary(title, summary):
    """Pretty-print a period summary table."""
    print('\n' + '-' * 70)
    print('%s - Mean +/- Std across seeds:' % title)
    print('-' * 70)
    print('%-8s %16s %16s %16s' % ('Model', 'TVM', 'MDD', 'Sharpe'))
    print('-' * 70)
    for model in summary.index:
        row = summary.loc[model]
        tvm_str = '%.4f +/- %.4f' % (row['tvm_mean'], row['tvm_std'])
        mdd_str = '%.4f +/- %.4f' % (row['mdd_mean'], row['mdd_std'])
        sr_str = '%.4f +/- %.4f' % (row['sharpe_mean'], row['sharpe_std'])
        print('%-8s %16s %16s %16s' % (model.upper(), tvm_str, mdd_str, sr_str))
    print('-' * 70)


def paired_t_tests(test_results, seeds, base='ppo', competitors=None):
    """Paired t-test of daily return differences (base - competitor) pooled across seeds."""
    from scipy import stats
    if competitors is None:
        competitors = [m for m in ['equal_weight', 'close', 'a2c', 'ddpg', 'td3', 'sac']
                       if m != base]

    rows = []
    for comp in competitors:
        diffs = []
        for s in seeds:
            df = test_results[s]
            if '%s_r' % base not in df.columns or '%s_r' % comp not in df.columns:
                continue
            d = (df['%s_r' % base] - df['%s_r' % comp]).dropna()
            if len(d) > 1:
                diffs.append(d)
        if not diffs:
            continue
        pooled = pd.concat(diffs)
        t, p = stats.ttest_1samp(pooled, 0)
        stars = '***' if p < 0.01 else ('**' if p < 0.05 else ('*' if p < 0.1 else ''))
        rows.append({
            'comparison': '%s vs %s' % (base, comp),
            'n_days': int(len(pooled)),
            'mean_daily_diff': pooled.mean(),
            't_stat': t,
            'p_value': p,
            'sig': stars,
        })
    return pd.DataFrame(rows)
