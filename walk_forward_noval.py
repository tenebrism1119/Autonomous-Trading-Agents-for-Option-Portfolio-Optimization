# -*- coding: utf-8 -*-
"""
Rolling out-of-sample backtest without a validation set.

Each window uses a 9-month training period followed by a 2-month test period and
contains no validation split; the model obtained at the end of training is used
directly. Windows slide forward by two months, giving seven consecutive windows
over the extended sample (2022-01 -- 2023-12).

Two of these windows correspond to the configurations reported in the paper:

  main    Section 4.2, Main experiment
          train 2022-07-01 ~ 2023-03-31 (184 trading days)
          test  2023-04-03 ~ 2023-05-31 (39 trading days)
  second  Section 4.3, Second rebalancing window
          train 2023-01-03 ~ 2023-09-28 (182 trading days)
          test  2023-10-09 ~ 2023-11-30 (39 trading days)

The remaining windows are produced by the same rolling procedure and are not
reported in the paper.

Protocol: five algorithms, three random seeds (42, 123, 456), equal-weight option
portfolio and buy-and-hold ETF benchmarks, pooled daily returns and paired t-tests.

CLI (window indices are 0-based; 3 = main experiment, 6 = second rebalancing window):
  python walk_forward_noval.py                  # full run (resumes, skips done windows)
  python walk_forward_noval.py --windows 3 6    # main experiment + second window
  python walk_forward_noval.py --summary-only   # pool existing results
  python walk_forward_noval.py --force          # ignore existing results

Output: results_wf_noval/  (<tag>_seed<seed>/, windows_metrics.csv,
        pooled_daily_returns.csv, pooled_metrics.csv, paired_t_test_wf.csv,
        nav_pooled.png, summary.txt)
"""

import os
import time
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from finrl.agents.stablebaselines3.models import DRLAgent

from pipeline import (
    WF_WINDOWS, N_WINDOWS, SEED, ALGOS, ALL_NAMES, DISPLAY,
    WF_TIMESTEPS, MODEL_CLS, MODEL_KWARGS,
    load_full_data, slice_period,
)
from config import ENV_KWARGS
from trading_env import StockTradingEnv
from evaluate import (
    build_benchmark, build_equal_weight_benchmark,
    max_drawdown, sharpe_ratio, compute_metrics,
)

NOVAL_DIR = 'results_wf_noval'

# 3 seeds per window: every window is trained once per seed, so a good/bad result
# cannot be attributed to a single lucky/unlucky run.
SEEDS = [42, 123, 456]

# Window corresponding to the main experiment of the paper.
MAIN_WINDOW_TAG = 'main'


# ---------------- training (NO validation) ----------------
def train_one(algo, train_data, out_dir, tag, seed):
    """Train one algorithm on the 9-month training slice; NO validation callback."""
    env_wrap = StockTradingEnv(df=train_data, **ENV_KWARGS)
    env_train, _ = env_wrap.get_sb_env()

    agent = DRLAgent(env=env_train)
    kw = MODEL_KWARGS[algo]
    if kw is not None:
        model = agent.get_model(algo, model_kwargs=kw, seed=seed)
    else:
        model = agent.get_model(algo, seed=seed)

    t0 = time.time()
    # NOTE: no ValidationSharpeCallback -> the FINAL model is kept (original scheme).
    model.learn(total_timesteps=WF_TIMESTEPS[algo],
                tb_log_name=f'{algo}_{tag}_noval_s{seed}')
    dt = time.time() - t0

    algo_dir = os.path.join(out_dir, algo)
    os.makedirs(algo_dir, exist_ok=True)
    model.save(os.path.join(algo_dir, f'agent_{algo}'))
    print(f'[Train] {algo.upper()} done ({tag}, seed={seed}, '
          f'{WF_TIMESTEPS[algo]} steps, {dt / 60:.1f} min, no validation)')
    return model


def ensure_models(win, train_data, out_dir, seed, force=False):
    models = {}
    for algo in ALGOS:
        final_path = os.path.join(out_dir, algo, f'agent_{algo}.zip')
        if os.path.exists(final_path) and not force:
            models[algo] = MODEL_CLS[algo].load(final_path)
            print(f'[Load] {algo.upper()} <- {final_path}')
        else:
            models[algo] = train_one(algo, train_data, out_dir, win['tag'], seed)
    return models


# ---------------- single window test-period evaluation ----------------
def evaluate_window(win, test_data, base_df, turnover_dates, out_dir, models):
    """Predict on the window's test period, save daily returns + metrics."""
    test_s, test_e_excl = win['test_s'], win['test_e_excl']
    n_days = test_data['date'].nunique()
    print(f'\n[Evaluate] {win["tag"]} test: [{test_s} ~ {test_e_excl}) '
          f'({n_days} days, {len(test_data)} rows)')

    e_test = StockTradingEnv(df=test_data, **ENV_KWARGS)
    av = {}
    for name, model in models.items():
        df_av, _ = DRLAgent.DRL_prediction(model=model, environment=e_test)
        av[name] = df_av

    # build_benchmark treats its end argument as INCLUSIVE; the test slice is
    # [test_s, test_e_excl) exclusive -> pass the LAST ACTUAL test date so the
    # benchmark covers exactly the same days as the model (avoids one extra day).
    bench = build_benchmark(base_df, test_s, str(test_data['date'].max()))
    ew = build_equal_weight_benchmark(
        test_data, turnover_dates, test_s, test_e_excl,
        initial_amount=ENV_KWARGS['initial_amount'])
    av['equal_weight'] = ew.reset_index()

    metrics, result = compute_metrics(av, bench)
    ret_cols = [f'{n}_r' for n in ALL_NAMES]
    rets = result[ret_cols].copy()
    rets.to_csv(os.path.join(out_dir, 'test_returns.csv'))
    metrics.to_csv(os.path.join(out_dir, 'metrics.csv'))
    print(f'[Eval] {win["tag"]} metrics:')
    print(metrics.to_string())
    return metrics


# ---------------- pooling / summary ----------------
def _collect_window_rets(seed):
    """{window_tag: test_returns_df} for one seed (missing windows skipped)."""
    rets = {}
    for win in WF_WINDOWS:
        p = os.path.join(NOVAL_DIR, f'{win["tag"]}_seed{seed}', 'test_returns.csv')
        if os.path.exists(p):
            rets[win['tag']] = pd.read_csv(p, index_col=0)
        else:
            print(f'[Warn] Missing {p}, skip {win["tag"]} (seed {seed})')
    return rets


def _pool_one_seed(all_rets):
    """Pool all windows of one seed into one daily-return DataFrame."""
    pooled = {}
    for name in ALL_NAMES:
        segs = [all_rets[tag][f'{name}_r'].dropna() for tag in all_rets]
        pooled[name] = pd.concat(segs).sort_index()
    df = pd.DataFrame(pooled)
    df.index = pd.to_datetime(df.index)
    return df[~df.index.duplicated(keep='last')]


def summarize():
    print('\n' + '=' * 60)
    print('Pooling all windows & final summary (NO validation, multi-seed)')
    print('=' * 60)

    seeds_used = [s for s in SEEDS
                  if any(os.path.exists(os.path.join(
                      NOVAL_DIR, f'{w["tag"]}_seed{s}', 'test_returns.csv'))
                      for w in WF_WINDOWS)]
    if not seeds_used:
        print('[Error] No window results found.')
        return
    print(f'[Seeds] {seeds_used}')

    win_rows, seed_rows, pooled_by_seed = [], [], {}
    for seed in seeds_used:
        all_rets = _collect_window_rets(seed)
        if not all_rets:
            continue
        for tag, rets in all_rets.items():
            for name in ALL_NAMES:
                x = rets[f'{name}_r'].dropna()
                win_rows.append({'window': tag, 'seed': seed, 'model': name,
                                 'tvm': (1 + x).prod(), 'mdd': max_drawdown(x),
                                 'sharpe': sharpe_ratio(x)})
        pdf = _pool_one_seed(all_rets)
        pooled_by_seed[seed] = pdf
        for name in ALL_NAMES:
            r = pdf[name].dropna()
            seed_rows.append({'seed': seed, 'model': name, 'tvm': (1 + r).prod(),
                              'mdd': max_drawdown(r), 'sharpe': sharpe_ratio(r)})

    win_df = pd.DataFrame(win_rows)
    win_df.to_csv(os.path.join(NOVAL_DIR, 'windows_metrics.csv'), index=False)
    seed_df = pd.DataFrame(seed_rows)
    seed_df.to_csv(os.path.join(NOVAL_DIR, 'seed_pooled_metrics.csv'), index=False)

    # ---- mean +/- std over seeds ----
    agg = seed_df.groupby('model').agg(
        tvm_mean=('tvm', 'mean'), tvm_std=('tvm', 'std'),
        mdd_mean=('mdd', 'mean'), mdd_std=('mdd', 'std'),
        sharpe_mean=('sharpe', 'mean'), sharpe_std=('sharpe', 'std')
    ).reindex(ALL_NAMES)
    disp = pd.DataFrame(index=ALL_NAMES)
    for c in ['tvm', 'mdd', 'sharpe']:
        disp[c] = (agg[f'{c}_mean'].round(4).astype(str) + ' +/- '
                   + agg[f'{c}_std'].round(4).astype(str))
    print('\n[Table] Pooled OOS metrics, mean +/- std over seeds:')
    print(disp.to_string())

    pooled_met = agg[['tvm_mean', 'mdd_mean', 'sharpe_mean']].rename(
        columns={'tvm_mean': 'tvm', 'mdd_mean': 'mdd', 'sharpe_mean': 'sharpe'})
    pooled_met.to_csv(os.path.join(NOVAL_DIR, 'pooled_metrics.csv'))
    agg.to_csv(os.path.join(NOVAL_DIR, 'pooled_summary.csv'))

    # ---- seed-averaged daily returns + NAV ----
    pooled_mean = sum(pooled_by_seed.values()) / len(pooled_by_seed)
    pooled_mean.to_csv(os.path.join(NOVAL_DIR, 'pooled_daily_returns.csv'))
    n_days = len(pooled_mean)
    print(f'\n[Pool] Pooled OOS days: {n_days} '
          f'({pooled_mean.index.min().date()} ~ {pooled_mean.index.max().date()})')

    print('\n[Table] Per-window TVM (mean over seeds):')
    print(win_df.pivot_table(index='window', columns='model', values='tvm',
                             aggfunc='mean')[ALL_NAMES].round(4).to_string())

    # ---- paired t-test per seed (PPO minus comparator, pooled within seed) ----
    trows = []
    for seed, pdf in pooled_by_seed.items():
        for comp in ['equal_weight', 'close', 'a2c', 'ddpg', 'td3', 'sac']:
            d = (pdf['ppo'] - pdf[comp]).dropna()
            t, p = stats.ttest_1samp(d, 0)
            trows.append({'seed': seed, 'comparison': f'ppo vs {comp}',
                          'n_days': int(len(d)), 'mean_daily_diff': d.mean(),
                          't_stat': t, 'p_value': p})
    ttest_df = pd.DataFrame(trows)
    ttest_df.to_csv(os.path.join(NOVAL_DIR, 'paired_t_test_wf.csv'), index=False)
    print('\n[Table] Paired t-tests by seed (pooled daily returns, PPO minus comparator):')
    print(ttest_df.round(5).to_string(index=False))

    # ---- pooled NAV figure: mean over seeds ----
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.figsize'] = (15, 6)
    fig, ax = plt.subplots()
    styles = {'ppo': dict(lw=2.2, color='tab:red', zorder=10),
              'equal_weight': dict(lw=1.6, color='tab:blue'),
              'close': dict(lw=1.6, color='tab:gray'),
              'a2c': dict(lw=1.0, ls='--'), 'ddpg': dict(lw=1.0, ls='--'),
              'td3': dict(lw=1.0, ls='--'), 'sac': dict(lw=1.0, ls='--')}
    nav = (1 + pooled_mean).cumprod()
    for name in ALL_NAMES:
        ax.plot(nav.index, nav[name].values, label=DISPLAY[name],
                **styles.get(name, {}))
    ax.axhline(1.0, color='black', lw=0.8, alpha=0.5)
    ax.legend()
    ax.set_ylabel('NAV (start = 1.0)')
    ax.set_title('Walk-Forward NAV, NO validation (9m train + 2m test), '
                 f'pooled {n_days} days, mean of seeds {seeds_used}')
    fig.tight_layout()
    fig.savefig(os.path.join(NOVAL_DIR, 'nav_pooled.png'), dpi=150)
    plt.close(fig)

    # ---- text summary ----
    lines = ['Walk-Forward Rolling Backtest (NO validation) Summary',
             '=' * 60,
             'Window size: train 9 months + test 2 months, no validation.',
             f'Sliding step: 2 months; Windows: {N_WINDOWS}; Seeds: {seeds_used}.',
             f'{MAIN_WINDOW_TAG} == main experiment in the paper (train 2022-07~2023-03, '
             'test 2023-04~2023-05).',
             f'Pooled out-of-sample days per seed: {n_days} '
             f'({pooled_mean.index.min().date()} ~ {pooled_mean.index.max().date()})',
             '', 'Pooled OOS metrics (mean +/- std over seeds):', disp.to_string(),
             '', 'Per-seed pooled metrics:', seed_df.round(4).to_string(index=False),
             '', 'Paired t-tests by seed (PPO minus comparator):',
             ttest_df.round(5).to_string(index=False)]
    with open(os.path.join(NOVAL_DIR, 'summary.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f'\n[Save] Summary: {NOVAL_DIR}/summary.txt')


def main():
    ap = argparse.ArgumentParser(description='Walk-forward backtest without validation set')
    ap.add_argument('--windows', nargs='+', type=int, default=None)
    ap.add_argument('--summary-only', action='store_true')
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()

    if args.summary_only:
        summarize()
        return

    os.makedirs(NOVAL_DIR, exist_ok=True)
    processed_df, expire_dates, turnover_dates, base_df = load_full_data()

    targets = args.windows if args.windows is not None else range(N_WINDOWS)
    t_start = time.time()
    for i in targets:
        win = WF_WINDOWS[i]
        for seed in SEEDS:
            tag = win['tag']
            out_dir = os.path.join(NOVAL_DIR, f'{tag}_seed{seed}')
            os.makedirs(out_dir, exist_ok=True)
            rets_path = os.path.join(out_dir, 'test_returns.csv')

            if os.path.exists(rets_path) and not args.force:
                print(f'\n[Skip] {tag} seed={seed} already done ({rets_path})')
                continue

            marker = '   <== MAIN EXPERIMENT' if tag == MAIN_WINDOW_TAG else ''
            print('\n' + '#' * 60)
            print('# Window %s (seed %d): train[%s ~ %s)  test[%s ~ %s)  '
                  '(NO validation)%s'
                  % (tag, seed, win['train_s'], win['test_s'], win['test_s'],
                     win['test_e_excl'], marker))
            print('#' * 60)
            w_t0 = time.time()

            train_data = slice_period(processed_df, win['train_s'], win['test_s'])
            test_data = slice_period(processed_df, win['test_s'], win['test_e_excl'])
            print(f'[Data] train={train_data["date"].nunique()}d, '
                  f'test={test_data["date"].nunique()}d (no validation)')

            models = ensure_models(win, train_data, out_dir, seed, force=args.force)
            evaluate_window(win, test_data, base_df, turnover_dates, out_dir, models)

            print(f'[Window {tag} seed={seed}] elapsed '
                  f'{(time.time() - w_t0) / 60:.1f} min, '
                  f'total {(time.time() - t_start) / 3600:.2f} h')

    summarize()


if __name__ == '__main__':
    main()
