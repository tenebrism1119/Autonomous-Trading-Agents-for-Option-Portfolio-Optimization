# -*- coding: utf-8 -*-
"""
Extended-sample experiment without a validation set.

Purpose
-------
The paper reports three configurations, all under the same validation-free
protocol: for every algorithm and every seed the model obtained at the end of
training is kept, with no checkpoint selection.

This script produces the extended-sample configuration (Section 4.4) with a
two-way split:

    train = [2022-01-01, 2023-04-01)   (date <  TRAIN_SPLIT_DATE)
    test  = [2023-04-01, 2023-12-31]   (date >= TRAIN_SPLIT_DATE)
    NO validation set, final model kept (no checkpoint selection).

Everything else is kept identical to the base pipeline (same data pipeline,
features, environment, 5 algorithms, TRAIN_TIMESTEPS, 3 seeds 42/123/456,
evaluation and t-test logic).

All outputs go to `results_ext_noval/` so that other result folders are never
touched.

CLI:
  python extended_noval.py                # full run, 3 seeds (resumes done seeds)
  python extended_noval.py --force        # ignore existing models, retrain all
  python extended_noval.py --seeds 42 123 # custom seeds
"""

import os
import time
import json
import argparse

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: F401
from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3
from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.meta.preprocessor.preprocessors import FeatureEngineer

from config import (
    ENV_KWARGS, START_DATE, TRAIN_SPLIT_DATE, ALGORITHMS, SEEDS,
    TRAIN_TIMESTEPS, PPO_PARAMS, TD3_PARAMS, SAC_PARAMS,
    DATA_SAVE_DIR, TRAINED_MODEL_DIR, TENSORBOARD_LOG_DIR,
)
from data_loader import calculate_turbulence, calculate_hv, calculate_bs_greeks
from trading_env import StockTradingEnv
from evaluate import (
    predict_all, build_benchmark, build_equal_weight_benchmark,
    compute_metrics,
)
from pipeline import (
    step1_data, check_and_make_directories,
    _period_summary, _print_summary, paired_t_tests,
)

OUT_DIR = 'results_ext_noval'
MODEL_CLS = {'a2c': A2C, 'ddpg': DDPG, 'ppo': PPO, 'td3': TD3, 'sac': SAC}
MODEL_KW = {'a2c': None, 'ddpg': None, 'ppo': PPO_PARAMS,
            'td3': TD3_PARAMS, 'sac': SAC_PARAMS}


# ---------------- data: two-way split (NO validation) ----------------
def build_data():
    """Load data + features, then split train/test WITHOUT a validation set."""
    print('\n' + '=' * 60)
    print('Data Loading & Feature Engineering (two-way split, NO validation)')
    print('=' * 60)
    hs300_df, hv_df, expire_dates, turnover_dates, base_df = step1_data()

    hs300_df = calculate_bs_greeks(hs300_df)
    fe = FeatureEngineer(use_technical_indicator=False,
                         use_turbulence=False,
                         user_defined_feature=False)
    processed_df = fe.preprocess_data(hs300_df)

    turbulence_index = calculate_turbulence(hs300_df)
    turbulence_index['turbulence'] = 0
    turbulence_index['turbulence'] = turbulence_index.apply(
        lambda x: 100 if x.date in expire_dates else 0, axis=1)

    processed_df = processed_df.merge(turbulence_index, on='date')
    processed_df = processed_df.merge(hv_df, on='date')
    processed_df.index = [int(i / 6) for i in list(processed_df.index)]

    train_data = processed_df.loc[processed_df.date < TRAIN_SPLIT_DATE, :]
    test_data = processed_df.loc[processed_df.date >= TRAIN_SPLIT_DATE, :]
    train_data.index = train_data['date'].factorize()[0]
    test_data.index = test_data['date'].factorize()[0]

    print(f'  Train: {len(train_data)} rows, {train_data["date"].nunique()} days '
          f'({train_data["date"].min()} ~ {train_data["date"].max()})')
    print(f'  Test:  {len(test_data)} rows, {test_data["date"].nunique()} days '
          f'({test_data["date"].min()} ~ {test_data["date"].max()})')
    print('  Validation: NONE')
    return train_data, test_data, base_df, turnover_dates


# ---------------- training (NO validation, final model kept) ----------------
def train_one(algo, env_train, seed, force=False):
    """Train one algorithm; save to results_ext_noval/run_seed{seed}/{algo}/."""
    save_dir = os.path.join(OUT_DIR, f'run_seed{seed}', algo)
    final_zip = os.path.join(save_dir, f'agent_{algo}.zip')
    if os.path.exists(final_zip) and not force:
        print(f'[Load] {algo.upper()} <- {final_zip}')
        return MODEL_CLS[algo].load(final_zip)

    agent = DRLAgent(env=env_train)
    kw = MODEL_KW[algo]
    if kw is None:
        model = agent.get_model(algo, seed=seed)
    else:
        model = agent.get_model(algo, model_kwargs=kw, seed=seed)

    t0 = time.time()
    # NO ValidationSharpeCallback -> the FINAL model is kept (original single-window scheme).
    model.learn(total_timesteps=TRAIN_TIMESTEPS[algo],
                tb_log_name=f'{algo}_extnoval_s{seed}')
    dt = time.time() - t0

    os.makedirs(save_dir, exist_ok=True)
    model.save(os.path.join(save_dir, f'agent_{algo}'))
    print(f'[Train] {algo.upper()} done (seed={seed}, {TRAIN_TIMESTEPS[algo]} steps, '
          f'{dt / 60:.1f} min, no validation)')
    return model


# ---------------- evaluation for one period ----------------
def eval_period(models, env, base_df, start_date, end_date, data,
                turnover_dates, tag, seed):
    """Predict + metrics + NAV plot for one period, saved under OUT_DIR."""
    av, actions = predict_all(models, env)
    bench = build_benchmark(base_df, start_date, end_date,
                            ENV_KWARGS['initial_amount'])
    ew = build_equal_weight_benchmark(
        data, turnover_dates, start_date, end_date,
        initial_amount=ENV_KWARGS['initial_amount'])
    av['equal_weight'] = ew.reset_index()

    metrics, result = compute_metrics(av, bench)

    run_dir = os.path.join(OUT_DIR, f'run_seed{seed}')
    os.makedirs(run_dir, exist_ok=True)
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    fig, ax = plt.subplots(figsize=(15, 5))
    if 'close' in result.columns:
        ax.plot(result['close'].values, label='Buy & Hold (ETF)', linewidth=1.5)
    markers = {'a2c': '1', 'ddpg': '2', 'ppo': '+', 'td3': '*', 'sac': 'o'}
    for name in models.keys():
        if name in result.columns:
            ax.plot(result[name].values, '--', label=name.upper(),
                    linewidth=0.8, marker=markers.get(name, ''))
    if 'equal_weight' in result.columns:
        ax.plot(result['equal_weight'].values, ':', label='EQUAL_WEIGHT', linewidth=1.2)
    ax.legend()
    ax.set_ylabel('NAV')
    ax.set_title(f'{tag.capitalize()} NAV Comparison (no validation, seed={seed})')
    plt.tight_layout()
    fig.savefig(os.path.join(run_dir, f'nav_{tag}.png'), dpi=150)
    plt.close(fig)

    result.to_csv(os.path.join(run_dir, f'returns_{tag}.csv'))
    metrics.to_csv(os.path.join(run_dir, f'metrics_{tag}.csv'))
    print(f'[Eval] {tag} metrics (seed={seed}):')
    print(metrics.round(4).to_string())
    return metrics, result


# ---------------- main ----------------
def main():
    parser = argparse.ArgumentParser(description='Extended dataset, NO validation')
    parser.add_argument('--force', action='store_true',
                        help='Ignore existing models and retrain')
    parser.add_argument('--seeds', nargs='+', type=int, default=None)
    args = parser.parse_args()
    seeds = args.seeds if args.seeds else SEEDS

    check_and_make_directories([DATA_SAVE_DIR, TRAINED_MODEL_DIR,
                                TENSORBOARD_LOG_DIR, OUT_DIR])

    train_data, test_data, base_df, turnover_dates = build_data()

    train_start = START_DATE
    train_end = str(train_data['date'].max())
    test_start = str(test_data['date'].min())
    test_end = str(test_data['date'].max())

    all_test_metrics, all_train_metrics, all_test_results = {}, {}, {}
    t_start = time.time()

    for seed in seeds:
        print('\n' + '#' * 60)
        print(f'# Run with seed={seed}  (NO validation)')
        print('#' * 60)

        e_train = StockTradingEnv(df=train_data, **ENV_KWARGS)
        e_test = StockTradingEnv(df=test_data, **ENV_KWARGS)
        env_train, _ = e_train.get_sb_env()

        models = {}
        for algo in ALGORITHMS:
            models[algo] = train_one(algo, env_train, seed, force=args.force)

        test_metrics, test_result = eval_period(
            models, e_test, base_df, test_start, test_end,
            test_data, turnover_dates, 'test', seed)
        train_metrics, _ = eval_period(
            models, e_train, base_df, train_start, train_end,
            train_data, turnover_dates, 'train', seed)

        all_test_metrics[seed] = test_metrics
        all_train_metrics[seed] = train_metrics
        all_test_results[seed] = test_result
        print(f'[Seed {seed}] elapsed {(time.time() - t_start) / 60:.1f} min')

    # ---- multi-seed summary + paired t-tests ----
    print('\n' + '=' * 60)
    print('Multi-Seed Summary (mean ± std) + Paired t-Tests (NO validation)')
    print('=' * 60)
    model_names = ALGORITHMS + ['equal_weight', 'close']
    test_summary = _period_summary(all_test_metrics, seeds, model_names)
    train_summary = _period_summary(all_train_metrics, seeds, model_names)
    _print_summary('Out-of-Sample (Test Set, no validation)', test_summary)
    _print_summary('In-Sample (Train Set, no validation)', train_summary)

    ttest_df = paired_t_tests(all_test_results, seeds)
    print('\n' + '-' * 70)
    print('Paired t-tests (test set daily returns, pooled across seeds):')
    print('-' * 70)
    if not ttest_df.empty:
        print(ttest_df.to_string(index=False))
    print('-' * 70)

    test_summary.to_csv(os.path.join(OUT_DIR, 'multi_seed_summary_test.csv'))
    train_summary.to_csv(os.path.join(OUT_DIR, 'multi_seed_summary_train.csv'))
    if not ttest_df.empty:
        ttest_df.to_csv(os.path.join(OUT_DIR, 'paired_t_test.csv'), index=False)
    raw = {
        'config': {
            'start': START_DATE, 'test_split': TRAIN_SPLIT_DATE,
            'validation': 'NONE', 'seeds': seeds,
            'train_timesteps': TRAIN_TIMESTEPS,
            'note': 'extended dataset, two-way split, no validation '
                    '(no validation split)',
        },
        'test': {str(s): all_test_metrics[s].to_dict() for s in seeds},
        'train': {str(s): all_train_metrics[s].to_dict() for s in seeds},
    }
    with open(os.path.join(OUT_DIR, 'multi_seed_raw.json'), 'w',
              encoding='utf-8') as f:
        json.dump(raw, f, indent=2, ensure_ascii=False)

    print(f'\n[Save] All outputs -> {OUT_DIR}/')
    print('=' * 60)
    print(f'Pipeline complete! Total {(time.time() - t_start) / 3600:.2f} h')


if __name__ == '__main__':
    main()
