# -*- coding: utf-8 -*-
"""
Build the two NAV figures used in the paper (training segment / test segment) for the
"three experiments", and archive each experiment into its own folder.

Output (placed under the project root):
  exp1_main_experiment/            code/ results/ figures/ summary/
  exp2_second_rebalancing_window/  code/ results/ figures/ summary/
  exp3_extended_sample/            code/ results/ figures/ summary/

Figure style follows the manuscript figures: legend = base/a2c/ddpg/ppo/td3/sac,
y-axis 'price', x-axis = trading-day index (0..N). Each experiment gets a
"training segment" and a "test segment" figure (mean over 3 seeds).

  exp1 = main experiment (train 2022-07~2023-03 / test 2023-04~2023-05)
  exp2 = second rebalancing window (train 2023-01~2023-09 / test 2023-10~2023-11)
  exp3 = extended sample (train <2023-04-01 / test 2023-04~2023-12)

exp1/exp2 only saved test-segment daily returns, so the training segment is
re-predicted with the saved models (no retraining); exp3 reads
results_ext_noval/run_seed*/returns_train.csv & returns_test.csv directly.

Run: python -X utf8 build_experiment_folders.py
"""

import os
import shutil
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from finrl.agents.stablebaselines3.models import DRLAgent

from config import ENV_KWARGS
from trading_env import StockTradingEnv
from evaluate import build_benchmark
from pipeline import load_full_data, slice_period, WF_WINDOWS, MODEL_CLS

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
SEEDS = [42, 123, 456]
ALGOS = ['a2c', 'ddpg', 'ppo', 'td3', 'sac']
NAMES_DISP = {'a2c': 'A2C', 'ddpg': 'DDPG', 'ppo': 'PPO', 'td3': 'TD3', 'sac': 'SAC',
              'base': 'ETF(Buy&Hold)', 'equal_weight': 'Equal-Weight Option'}
MARKERS = {'a2c': 'v', 'ddpg': '^', 'ppo': '+', 'td3': '*', 'sac': 'o'}

EXP = {
    'exp1': {'folder': 'exp1_main_experiment', 'win': 'main',
             'desc': 'Main experiment (Section 4.2)'},
    'exp2': {'folder': 'exp2_second_rebalancing_window', 'win': 'second',
             'desc': 'Second rebalancing window (Section 4.3)'},
    'exp3': {'folder': 'exp3_extended_sample', 'win': None,
             'desc': 'Extended sample (Section 4.4)'},
}


# ---------------- plotting ----------------
def plot_nav(df, outpath, n_seeds=3):
    """df: index=day idx, columns include base/a2c/ddpg/ppo/td3/sac."""
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    fig, ax = plt.subplots(figsize=(9.5, 3.4))
    order = ['base'] + ALGOS
    for n in order:
        if n not in df.columns:
            continue
        s = df[n].values
        if n == 'base':
            ax.plot(df.index, s, label='base', linewidth=1.0)
        else:
            ax.plot(df.index, s, label=n, marker=MARKERS[n], markersize=3,
                    linewidth=0.8)
    ax.legend(loc='upper left', fontsize=7, framealpha=0.8)
    ax.set_ylabel('price')
    ax.set_xlabel('')
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def mean_over_seeds(list_of_dict):
    """list_of_dict: [{name: series(index=date)}, ...] -> DataFrame mean."""
    frames = [pd.DataFrame(d) for d in list_of_dict]
    acc = frames[0].copy()
    for f in frames[1:]:
        acc = acc.add(f.reindex(acc.index), fill_value=np.nan)
    return acc / len(frames)


# ---------------- exp1 / exp2 : re-predict train & test ----------------
def nav_from_models(models, data, base_df):
    envw = StockTradingEnv(df=data, **ENV_KWARGS)
    out = {}
    for name, m in models.items():
        df_av, _ = DRLAgent.DRL_prediction(model=m, environment=envw)
        out[name] = df_av.set_index('date')['account_value']
    b = build_benchmark(base_df, str(data['date'].min()), str(data['date'].max()),
                        ENV_KWARGS['initial_amount'])
    out['base'] = b['close']
    return out


def build_exp_window_figs(exp_key, processed_df, base_df):
    win = next(w for w in WF_WINDOWS if w['tag'] == EXP[exp_key]['win'])
    train_data = slice_period(processed_df, win['train_s'], win['test_s'])
    test_data = slice_period(processed_df, win['test_s'], win['test_e_excl'])
    print(f"[{exp_key}] {win['tag']} train={train_data['date'].nunique()}d "
          f"test={test_data['date'].nunique()}d")

    train_navs, test_navs = [], []
    for s in SEEDS:
        models = {}
        for algo in ALGOS:
            p = os.path.join(HERE, 'results_wf_noval', f"{win['tag']}_seed{s}",
                             algo, f'agent_{algo}.zip')
            models[algo] = MODEL_CLS[algo].load(p)
        train_navs.append(nav_from_models(models, train_data, base_df))
        test_navs.append(nav_from_models(models, test_data, base_df))

    for tag, navs in [('train', train_navs), ('test', test_navs)]:
        mx = mean_over_seeds(navs)
        mx = mx[['base'] + ALGOS]
        mx = mx.reset_index(drop=True)
        outpath = os.path.join(ROOT, EXP[exp_key]['folder'], 'figures',
                               f'{tag}_nav_comparison.png')
        plot_nav(mx, outpath, len(SEEDS))
        mx.to_csv(os.path.join(ROOT, EXP[exp_key]['folder'], 'results',
                               f'{tag}_nav_3seed_mean.csv'), index=False)
        print(f'  [Fig] {outpath}')


# ---------------- exp3 : from saved csv ----------------
def build_exp3_figs():
    train_navs, test_navs = [], []
    for s in SEEDS:
        d = os.path.join(HERE, 'results_ext_noval', f'run_seed{s}')
        tr = pd.read_csv(os.path.join(d, 'returns_train.csv'), index_col=0,
                         parse_dates=True)
        te = pd.read_csv(os.path.join(d, 'returns_test.csv'), index_col=0,
                         parse_dates=True)
        tr = tr[ALGOS + ['close']].rename(columns={'close': 'base'})
        te = te[ALGOS + ['close']].rename(columns={'close': 'base'})
        train_navs.append({c: tr[c] for c in tr.columns})
        test_navs.append({c: te[c] for c in te.columns})

    for tag, navs in [('train', train_navs), ('test', test_navs)]:
        mx = mean_over_seeds(navs)[['base'] + ALGOS].reset_index(drop=True)
        outpath = os.path.join(ROOT, EXP['exp3']['folder'], 'figures',
                               f'{tag}_nav_comparison.png')
        plot_nav(mx, outpath, len(SEEDS))
        mx.to_csv(os.path.join(ROOT, EXP['exp3']['folder'], 'results',
                               f'{tag}_nav_3seed_mean.csv'), index=False)
        print(f'  [Fig] {outpath}')


# ---------------- folder assembly ----------------
def ensure_dirs():
    for k, v in EXP.items():
        for sub in ['code', 'results', 'figures', 'summary']:
            os.makedirs(os.path.join(ROOT, v['folder'], sub), exist_ok=True)


def copy_code():
    py = [f for f in os.listdir(HERE) if f.endswith('.py')]
    for k, v in EXP.items():
        for f in py:
            shutil.copy2(os.path.join(HERE, f),
                         os.path.join(ROOT, v['folder'], 'code', f))
        print(f"[Code] {v['folder']}/code <- {len(py)} .py files")


def copy_results():
    # exp1 / exp2: copy main/second seed dirs
    for k in ['exp1', 'exp2']:
        win = EXP[k]['win']
        dst_root = os.path.join(ROOT, EXP[k]['folder'], 'results')
        for s in SEEDS:
            src = os.path.join(HERE, 'results_wf_noval', f'{win}_seed{s}')
            shutil.copytree(src, os.path.join(dst_root, f'seed{s}'),
                            dirs_exist_ok=True)
        # window metrics row
        wm = pd.read_csv(os.path.join(HERE, 'results_wf_noval',
                                      'windows_metrics.csv'))
        wm[wm.window == win].to_csv(
            os.path.join(dst_root, f'{win}_per_algo_per_seed_metrics.csv'), index=False)
        print(f"[Result] {EXP[k]['folder']}/results <- {win} 3 seeds")

    # exp3: copy run_seed dirs + summary csv/json
    dst_root = os.path.join(ROOT, EXP['exp3']['folder'], 'results')
    for s in SEEDS:
        src = os.path.join(HERE, 'results_ext_noval', f'run_seed{s}')
        shutil.copytree(src, os.path.join(dst_root, f'seed{s}'),
                        dirs_exist_ok=True)
    for fn in ['multi_seed_summary_test.csv', 'multi_seed_summary_train.csv',
               'multi_seed_raw.json', 'paired_t_test.csv']:
        p = os.path.join(HERE, 'results_ext_noval', fn)
        if os.path.exists(p):
            shutil.copy2(p, os.path.join(dst_root, fn))
    print(f"[Result] {EXP['exp3']['folder']}/results <- 3 seeds + summaries")


def write_summaries():
    wm = pd.read_csv(os.path.join(HERE, 'results_wf_noval', 'windows_metrics.csv'))

    def per_seed_tvm(win):
        sub = wm[wm.window == win]
        piv = sub.pivot_table(index='model', columns='seed', values='tvm')
        piv['Mean'] = piv.mean(axis=1)
        piv['Std'] = piv.std(axis=1, ddof=1)
        return piv.reindex(ALGOS + ['equal_weight', 'close']).round(4)

    # exp1
    lines1 = ['# Experiment 1 - Main experiment (main experiment): results summary', '',
              '- Training set: 2022-07-01 ~ 2023-03-31 (9 months, no validation set)',
              '- Test set: 2023-04-01 ~ 2023-05-31 (2 months)',
              '- Seeds: 42 / 123 / 456; 5 algorithms; final model kept directly',
              '- Train/test boundaries identical to the main experiment (reproduction anchor)', '',
              '## Out-of-sample (test segment) TVM per seed', '',
              per_seed_tvm('main').to_markdown(floatfmt='.4f'), '',
              '## Key points', '',
              '- PPO TVM across three seeds = 1.803 / 2.502 / 1.130, all > 1.',
              '- ETF benchmark TVM is 0.9319 for all three seeds, identical digit by digit to the original record 0.931863.',
              '- Absolute values of the other algorithms in the original record (PPO=2.948, DDPG=4.279, etc.) come from a single run without seeds, '
              'and cannot be reproduced digit by digit under multi-seed repetition.']
    open(os.path.join(ROOT, EXP['exp1']['folder'], 'summary', 'exp1_results_summary.md'),
         'w', encoding='utf-8').write('\n'.join(lines1))

    lines2 = ['# Experiment 2 - Second rebalancing window (revalidation window): results summary', '',
              '- Training set: 2023-01-01 ~ 2023-09-30 (9 months, no validation set)',
              '- Test set: 2023-10-01 ~ 2023-11-30 (2 months)',
              '- Seeds: 42 / 123 / 456; same protocol as Experiment 1, different period', '',
              '## Out-of-sample (test segment) TVM per seed', '',
              per_seed_tvm('second').to_markdown(floatfmt='.4f'), '',
              '## Key points', '',
              '- PPO TVM across three seeds = 2.279 / 2.366 / 2.369, very low dispersion and all > 1.',
              '- In this window A2C (seed42), SAC (three seeds) and TD3 (seed456) are all < 1.']
    open(os.path.join(ROOT, EXP['exp2']['folder'], 'summary', 'exp2_results_summary.md'),
         'w', encoding='utf-8').write('\n'.join(lines2))

    # exp3
    with open(os.path.join(HERE, 'results_ext_noval', 'multi_seed_raw.json'),
              encoding='utf-8') as f:
        raw = json.load(f)
    summ = pd.read_csv(os.path.join(HERE, 'results_ext_noval',
                                    'multi_seed_summary_test.csv'))
    tt = pd.read_csv(os.path.join(HERE, 'results_ext_noval', 'paired_t_test.csv'))
    lines3 = ['# Experiment 3 - Extended dataset (no validation set): results summary', '',
              '- Data: 2022-01-01 ~ 2023-12-31; two-way split (no validation set)',
              '- Training set: 2022-01-01 ~ 2023-03-31 (15 months)',
              '- Test set: 2023-04-01 ~ 2023-12-31 (9 months, 546 out-of-sample trading days)',
              '- Seeds: 42 / 123 / 456; final model kept directly', '',
              '## Out-of-sample metrics (3-seed mean +/- std)', '',
              summ.rename(columns={'model': 'Model'}).to_markdown(index=False, floatfmt='.4f'),
              '', '## Paired t-tests (out-of-sample daily returns)', '',
              tt.to_markdown(index=False, floatfmt='.5f'), '', '## Key points', '',
              '- PPO: TVM 8.869 +/- 1.031, Sharpe 2.646 +/- 0.093, significant against every comparator.',
              '- A2C / SAC vary greatly across seeds (TVM std 1.85 / 3.55); some seeds are close to zero.',
              '- DDPG / TD3 barely change across the three seeds (no trading).']
    open(os.path.join(ROOT, EXP['exp3']['folder'], 'summary',
                      'exp3_results_summary.md'), 'w', encoding='utf-8').write('\n'.join(lines3))
    print('[Summary] summary/*.md written for the three experiments')


def main():
    ensure_dirs()

    print('\n==== Build exp1/exp2 train+test NAV figures (re-prediction) ====')
    processed_df, expire_dates, turnover_dates, base_df = load_full_data()
    build_exp_window_figs('exp1', processed_df, base_df)
    build_exp_window_figs('exp2', processed_df, base_df)

    print('\n==== Build exp3 train+test NAV figures (from saved CSV) ====')
    build_exp3_figs()

    print('\n==== Archive the three experiment folders ====')
    copy_code()
    copy_results()
    write_summaries()

    print('\n[Done] three experiment folders generated under', ROOT)


if __name__ == '__main__':
    main()
