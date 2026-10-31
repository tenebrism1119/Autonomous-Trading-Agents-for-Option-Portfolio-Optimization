# -*- coding: utf-8 -*-
"""Redraw paper NAV figures with the equal-weight option portfolio benchmark.

Figures (all currently show only the buy-and-hold ETF benchmark):
  fig1.png                  main experiment, training segment (3-seed mean)
  fig2.png                  main experiment, test segment (3-seed mean)
  fig_insample_seeds.png    main experiment, training segment, one panel per seed
  fig_second_train.png      second rebalancing window, training segment
  fig_second_test.png       second rebalancing window, test segment
  fig_ext_train.png         extended sample, training segment
  fig_ext_test.png          extended sample, test segment

DRL / ETF curves come from the stored result CSVs; per-seed training NAVs of
the main experiment are reproduced from the saved agents.  The equal-weight
benchmark is computed with evaluate.build_equal_weight_benchmark, i.e. the
same routine that produces the benchmark rows of the paper's tables.
"""
import os
import shutil

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
# Paper project root: the parent of this code directory. Override with PAPER_ROOT.
PAPER_ROOT = os.environ.get('PAPER_ROOT', os.path.dirname(HERE))
# Output directory for the redrawn figures. Override with FIG_OUT_DIR.
OUT = os.environ.get('FIG_OUT_DIR', os.path.join(PAPER_ROOT, 'figures'))
# Archived result folders of the three experiments (individually overridable).
EXP = {
    'main': os.environ.get('EXP_MAIN_DIR',
                           os.path.join(PAPER_ROOT, 'exp1_main_experiment', 'results')),
    'second': os.environ.get('EXP_SECOND_DIR',
                             os.path.join(PAPER_ROOT, 'exp2_second_rebalancing_window', 'results')),
    'ext': os.environ.get('EXP_EXT_DIR',
                          os.path.join(PAPER_ROOT, 'exp3_extended_sample', 'results')),
}
SEEDS = [42, 123, 456]
ALGOS = ['a2c', 'ddpg', 'ppo', 'td3', 'sac']
INIT = 60000.0

# ---------------------------------------------------------------- plotting
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'mathtext.fontset': 'stix',
    'font.size': 11,
    'axes.linewidth': 1.0,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'legend.frameon': False,
    'savefig.dpi': 300,
})

ORDER = ['base', 'equal_weight', 'a2c', 'ddpg', 'ppo', 'td3', 'sac']
STYLE = {
    'base':         dict(color='#7f7f7f', ls=(0, (4, 3)), lw=1.4, label='Buy-and-hold ETF'),
    'equal_weight': dict(color='#000000', ls=(0, (1, 1.6)), lw=1.6,
                         label='Equal-weight option portfolio'),
    'a2c':          dict(color='#1f77b4', ls='-', lw=1.3, label='A2C'),
    'ddpg':         dict(color='#ff7f0e', ls='-', lw=1.3, label='DDPG'),
    'ppo':          dict(color='#d62728', ls='-', lw=3.0, label='PPO'),
    'td3':          dict(color='#2ca02c', ls='-', lw=1.3, label='TD3'),
    'sac':          dict(color='#9467bd', ls='-', lw=1.3, label='SAC'),
}


def draw(ax, series):
    """series: {name: 1-d array}."""
    for n in ORDER:
        if n in series:
            st = dict(STYLE[n])
            lab = st.pop('label')
            ax.plot(series[n], label=lab, **st)


def save_single(series, fname):
    fig, ax = plt.subplots(figsize=(10, 4.75))
    draw(ax, series)
    ax.set_ylabel('Portfolio value (CNY)')
    ax.set_xlabel('Trading day')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.16), ncol=7,
              fontsize=10.5, handlelength=2.2, columnspacing=1.4)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, fname), bbox_inches='tight')
    plt.close(fig)
    print('  saved', fname)


def save_seed_panels(panels, fname):
    """panels: [(seed, series), ...]"""
    fig, axes = plt.subplots(len(panels), 1, figsize=(10, 9.2), sharex=True)
    for ax, (seed, series) in zip(axes, panels):
        draw(ax, series)
        ax.set_ylabel('Portfolio value (CNY)')
        ax.set_title('Seed %d' % seed, loc='left', fontsize=11, pad=4)
    axes[-1].set_xlabel('Trading day')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=7, fontsize=10.5,
               bbox_to_anchor=(0.5, 0.0), handlelength=2.2, columnspacing=1.4)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(os.path.join(OUT, fname), bbox_inches='tight')
    plt.close(fig)
    print('  saved', fname)


# ---------------------------------------------------------------- data
def read_mean(bucket, seg):
    """3-seed-mean NAV from the stored CSV (base + 5 algos)."""
    df = pd.read_csv(os.path.join(EXP[bucket], f'{seg}_净值_3种子均值.csv'))
    return {c: df[c].values.astype(float) for c in df.columns}


def read_seed_returns(bucket, seg, tag):
    """Per-seed NAV from returns CSVs -> 3-seed mean dict."""
    frames = []
    for s in SEEDS:
        p = os.path.join(EXP[bucket], f'种子{s}', tag)
        df = pd.read_csv(p, index_col=0)
        nav = {}
        for c in ['a2c', 'ddpg', 'ppo', 'td3', 'sac', 'equal_weight', 'close']:
            if c in df.columns:
                nav[c] = df[c].astype(float).values
            elif f'{c}_r' in df.columns:
                nav[c] = (1 + df[f'{c}_r'].astype(float)).cumprod().values * INIT
            elif f'{c}_c' in df.columns:
                nav[c] = df[f'{c}_c'].astype(float).values * INIT
        frames.append(pd.DataFrame(nav))
    mean = frames[0].copy()
    for f in frames[1:]:
        mean = mean + f
    mean = mean / len(frames)
    return {c: mean[c].values for c in mean.columns}


def read_ext_raw(seg, tag):
    """extended sample: CSV stores the raw NAV columns."""
    frames = []
    for s in SEEDS:
        df = pd.read_csv(os.path.join(EXP['ext'], f'种子{s}', tag), index_col=0)
        nav = {c: df[c].astype(float).values
               for c in ['a2c', 'ddpg', 'ppo', 'td3', 'sac', 'equal_weight', 'close']}
        frames.append(pd.DataFrame(nav))
    mean = frames[0].copy()
    for f in frames[1:]:
        mean = mean + f
    return {c: (mean / len(frames))[c].values for c in mean.columns}


def to_series(d):
    """Rename raw columns to plotting keys and append the equal-weight line."""
    base = d['close'] if 'close' in d else d['base']
    out = {'base': base}
    for a in ALGOS:
        out[a] = d[a]
    if 'equal_weight' in d:
        out['equal_weight'] = d['equal_weight']
    return out


def predict_main_train_series(processed_df):
    """Per-seed training NAV of the main window from the saved agents."""
    from trading_env import StockTradingEnv
    from pipeline import WF_WINDOWS, slice_period, MODEL_CLS
    from finrl.agents.stablebaselines3.models import DRLAgent
    from config import ENV_KWARGS

    win = next(w for w in WF_WINDOWS if w['tag'] == 'main')
    train_data = slice_period(processed_df, win['train_s'], win['test_s'])
    print('  main train slice: %d days' % train_data['date'].nunique())

    per_seed = []
    for s in SEEDS:
        navs = {}
        for a in ALGOS:
            p = os.path.join(EXP['main'], f'种子{s}', a, f'agent_{a}.zip')
            model = MODEL_CLS[a].load(p)
            env = StockTradingEnv(df=train_data.copy(), **ENV_KWARGS)
            av, _ = DRLAgent.DRL_prediction(model=model, environment=env)
            navs[a] = av['account_value'].values.astype(float)
        per_seed.append(navs)
        print('    seed %d done' % s)
    return per_seed


def load_data_and_ew():
    """Load option data and compute the equal-weight NAV of the six segments."""
    # Default to the repository-local data folder; override with DATA_ROOT.
    os.environ.setdefault(
        'DATA_ROOT', os.path.join(HERE, 'data', 'data'))
    import sys
    sys.path.insert(0, HERE)
    from pipeline import load_full_data
    from evaluate import build_equal_weight_benchmark

    processed_df, _, turnover_dates, _ = load_full_data()

    spans = {
        ('main', 'train'): ('2022-07-01', '2023-03-31'),
        ('main', 'test'): ('2023-04-03', '2023-05-31'),
        ('second', 'train'): ('2023-01-03', '2023-09-28'),
        ('second', 'test'): ('2023-10-09', '2023-11-30'),
    }
    ew = {}
    for key, (s, e) in spans.items():
        r = build_equal_weight_benchmark(processed_df, turnover_dates, s, e, INIT)
        ew[key] = r['account_value'].values.astype(float)
        print('  EW %s %s: %d days, TVM %.3f' % (key[0], key[1], len(ew[key]),
                                                 ew[key][-1] / INIT))
    return processed_df, ew


def main():
    backup = os.path.join(HERE, '_fig_backup')
    os.makedirs(backup, exist_ok=True)
    for f in ['fig1.png', 'fig2.png', 'fig_insample_seeds.png',
              'fig_second_train.png', 'fig_second_test.png',
              'fig_ext_train.png', 'fig_ext_test.png']:
        src = os.path.join(OUT, f)
        if os.path.exists(src) and not os.path.exists(os.path.join(backup, f)):
            shutil.copy2(src, os.path.join(backup, f))

    processed_df, ew = load_data_and_ew()

    # ---- main experiment ----
    tr = read_mean('main', '训练段')
    te = read_mean('main', '测试段')
    for d, key in [(tr, ('main', 'train')), (te, ('main', 'test'))]:
        assert len(d['base']) == len(ew[key]), (len(d['base']), len(ew[key]))
        d['equal_weight'] = ew[key]
    save_single(to_series(tr), 'fig1.png')
    save_single(to_series(te), 'fig2.png')

    # ---- per-seed training panels ----
    per_seed = predict_main_train_series(processed_df)
    chk = pd.DataFrame(per_seed[0])
    for p in per_seed[1:]:
        chk = chk + pd.DataFrame(p)
    chk = (chk / len(per_seed)).values
    print('  check: max |reproduced mean - stored mean| = %.1f CNY'
          % np.max(np.abs(chk - np.column_stack([tr[a] for a in ALGOS]))))
    panels = []
    for s, navs in zip(SEEDS, per_seed):
        series = {'base': tr['base'], 'equal_weight': ew[('main', 'train')]}
        series.update(navs)
        panels.append((s, series))
    save_seed_panels(panels, 'fig_insample_seeds.png')

    # ---- second window ----
    tr2 = read_mean('second', '训练段')
    te2 = read_mean('second', '测试段')
    for d, key in [(tr2, ('second', 'train')), (te2, ('second', 'test'))]:
        assert len(d['base']) == len(ew[key]), (len(d['base']), len(ew[key]))
        d['equal_weight'] = ew[key]
    save_single(to_series(tr2), 'fig_second_train.png')
    save_single(to_series(te2), 'fig_second_test.png')

    # ---- extended sample ----
    ext_tr = read_ext_raw('train', 'returns_train.csv')
    ext_te = read_ext_raw('test', 'returns_test.csv')
    save_single(to_series(ext_tr), 'fig_ext_train.png')
    save_single(to_series(ext_te), 'fig_ext_test.png')

    print('done')


if __name__ == '__main__':
    main()
