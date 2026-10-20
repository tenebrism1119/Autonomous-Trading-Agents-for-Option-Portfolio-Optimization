


"""
Main entry point for Soybean Meal Option Portfolio RL Trading.

Plan B: generalisation evidence - American commodity options

Pipeline:
  0. (first run) fetch soybean meal option data from Tushare
  1. Load data + select contracts
  2. Feature engineering (CRR Greeks: IV/Delta/Gamma/Vega/Theta/EEP/Moneyness) + train/test split
  3. Build the trading environment (79-dimensional state space)
  4. Train 5 DRL models (A2C/DDPG/PPO/TD3/SAC) x 3 seeds
  5. Predict + evaluate (TVM/MDD/Sharpe) + plot NAV curves
  6. Multi-seed summary: mean ± std

Usage:
  python main.py          # full pipeline (3 seeds)
  python main.py --skip-train  # skip training, load saved models
  python main.py --seeds 42 123  # custom seeds
  python main.py --fetch-only  # only fetch data from Tushare, no training
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split

from config import (
    DATA_SAVE_DIR, TRAINED_MODEL_DIR, TENSORBOARD_LOG_DIR, RESULTS_DIR,
    ENV_KWARGS, START_DATE, END_DATE, TRAIN_SPLIT_DATE, ALGORITHMS,
    SEEDS, INDICATORS, CONTRACT_MULTIPLIER,
)
from data_loader import prepare_data, calculate_turbulence, calculate_hv, calculate_crr_greeks
from trading_env import StockTradingEnv
from train import train_all
from evaluate import evaluate


def check_and_make_directories(directories):
    for d in directories:
        if not os.path.exists("./" + d):
            os.makedirs("./" + d)


def step1_data():
    print("\n" + "=" * 60)
    print("Step 1: Data Loading & Contract Selection (soybean meal options)")
    print("=" * 60)
    hs300_df, hv_df, expire_dates, turnover_dates, base_df = prepare_data()
    return hs300_df, hv_df, expire_dates, turnover_dates, base_df


def step2_features(hs300_df, hv_df, expire_dates):
    print("\n" + "=" * 60)
    print("Step 2: Feature Engineering & Data Split (CRR Binomial)")
    print("=" * 60)

    print("  Computing CRR binomial Greeks (this may take several minutes)...")
    hs300_df = calculate_crr_greeks(hs300_df)
    print(f"  CRR Greeks: iv/delta/gamma/vega/theta/eep/moneyness added")
    print(f"  State space: {1 + 2 * 6 + 11 * 6} (was 67 in Plan A, now 79)")

    feature_engineer = FeatureEngineer(
        use_technical_indicator=False,
        use_turbulence=False,
        user_defined_feature=False,
    )
    processed_df = feature_engineer.preprocess_data(hs300_df)

    turbulence_index = calculate_turbulence(hs300_df)
    turbulence_index['turbulence'] = 0
    turbulence_index['turbulence'] = turbulence_index.apply(
        lambda x: 100 if x.date in expire_dates else 0, axis=1
    )

    processed_df = processed_df.merge(turbulence_index, on='date')
    processed_df = processed_df.merge(hv_df, on='date')
    processed_df.index = [int(i / 6) for i in list(processed_df.index)]

    train_data = processed_df.loc[processed_df.date < TRAIN_SPLIT_DATE, :]
    test_data = processed_df.loc[processed_df.date >= TRAIN_SPLIT_DATE, :]
    test_data.index = test_data['date'].factorize()[0]

    print(f"  Train: {len(train_data)} rows, {train_data['date'].nunique()} days")
    print(f"  Test:  {len(test_data)} rows, {test_data['date'].nunique()} days")
    return train_data, test_data


def step3_env(train_data, test_data):
    print("\n" + "=" * 60)
    print("Step 3: Build Trading Environment")
    print("=" * 60)

    e_train = StockTradingEnv(df=train_data, **ENV_KWARGS)
    e_test = StockTradingEnv(df=test_data, **ENV_KWARGS)
    env_train, _ = e_train.get_sb_env()

    print(f"  Stock dim: {ENV_KWARGS['stock_dim']}")
    print(f"  State space: {ENV_KWARGS['state_space']}")
    print(f"  Initial amount: {ENV_KWARGS['initial_amount']}")
    print(f"  hmax: {ENV_KWARGS['hmax']}")
    print(f"  Contract multiplier: {CONTRACT_MULTIPLIER} (price unit: CNY/ton, no conversion)")
    return env_train, e_train, e_test


def step4_train(env_train, seed=42):
    print("\n" + "=" * 60)
    print(f"Step 4: Model Training (seed={seed})")
    print("=" * 60)
    models = train_all(env_train, seed=seed)
    return models


def step5_evaluate(models, e_train, e_test, base_df, seed=42,
                    train_data=None, test_data=None, turnover_dates=None):
    print("\n" + "=" * 60)
    print(f"Step 5: Prediction & Evaluation (seed={seed})")
    print("=" * 60)

    test_start = '2023-04-03'
    test_end = END_DATE
    train_start = START_DATE
    train_end = '2023-03-31'
    print(f"  Train: {train_start} ~ {train_end}")
    print(f"  Test:  {test_start} ~ {test_end}")

    test_metrics, train_metrics = evaluate(
        models=models,
        test_env=e_test,
        train_env=e_train,
        base_df=base_df,
        test_start=test_start,
        test_end=test_end,
        train_start=train_start,
        train_end=train_end,
        initial_amount=ENV_KWARGS['initial_amount'],
        seed=seed,
        test_data=test_data,
        train_data=train_data,
        turnover_dates=turnover_dates,
    )
    return test_metrics, train_metrics


def step6_summary(all_test_metrics, all_train_metrics, seeds):
    print("\n" + "=" * 60)
    print("Step 6: Multi-Seed Summary (mean ± std)")
    print("=" * 60)

    model_names = ALGORITHMS + ['equal_weight', 'close']

    test_rows = []
    for model in model_names:
        tvms = [float(all_test_metrics[s].loc[model, 'tvm']) for s in seeds]
        mdds = [float(all_test_metrics[s].loc[model, 'mdd']) for s in seeds]
        srs = [float(all_test_metrics[s].loc[model, 'sharpe']) for s in seeds]
        test_rows.append({
            'model': model,
            'tvm_mean': np.mean(tvms),
            'tvm_std': np.std(tvms, ddof=1) if len(tvms) > 1 else 0.0,
            'mdd_mean': np.mean(mdds),
            'mdd_std': np.std(mdds, ddof=1) if len(mdds) > 1 else 0.0,
            'sharpe_mean': np.mean(srs),
            'sharpe_std': np.std(srs, ddof=1) if len(srs) > 1 else 0.0,
        })
    test_summary = pd.DataFrame(test_rows).set_index('model')

    train_rows = []
    for model in model_names:
        tvms = [float(all_train_metrics[s].loc[model, 'tvm']) for s in seeds]
        mdds = [float(all_train_metrics[s].loc[model, 'mdd']) for s in seeds]
        srs = [float(all_train_metrics[s].loc[model, 'sharpe']) for s in seeds]
        train_rows.append({
            'model': model,
            'tvm_mean': np.mean(tvms),
            'tvm_std': np.std(tvms, ddof=1) if len(tvms) > 1 else 0.0,
            'mdd_mean': np.mean(mdds),
            'mdd_std': np.std(mdds, ddof=1) if len(mdds) > 1 else 0.0,
            'sharpe_mean': np.mean(srs),
            'sharpe_std': np.std(srs, ddof=1) if len(srs) > 1 else 0.0,
        })
    train_summary = pd.DataFrame(train_rows).set_index('model')

    print("\n" + "-" * 70)
    print("Out-of-Sample (Test Set) - Mean ± Std across seeds:")
    print("-" * 70)
    print(f"{'Model':<8} {'TVM':>16} {'MDD':>16} {'Sharpe':>16}")
    print("-" * 70)
    for model in model_names:
        row = test_summary.loc[model]
        tvm_str = f"{row['tvm_mean']:.4f} ± {row['tvm_std']:.4f}"
        mdd_str = f"{row['mdd_mean']:.4f} ± {row['mdd_std']:.4f}"
        sr_str = f"{row['sharpe_mean']:.4f} ± {row['sharpe_std']:.4f}"
        print(f"{model.upper():<8} {tvm_str:>16} {mdd_str:>16} {sr_str:>16}")
    print("-" * 70)

    test_summary.to_csv('results/multi_seed_summary_test.csv')
    train_summary.to_csv('results/multi_seed_summary_train.csv')
    print("\n[Save] Test summary:  results/multi_seed_summary_test.csv")
    print("[Save] Train summary: results/multi_seed_summary_train.csv")

    raw = {
        'test': {str(s): all_test_metrics[s].to_dict() for s in seeds},
        'train': {str(s): all_train_metrics[s].to_dict() for s in seeds},
    }
    with open('results/multi_seed_raw.json', 'w') as f:
        json.dump(raw, f, indent=2)
    print("[Save] Raw results:   results/multi_seed_raw.json")

    return test_summary, train_summary


def main():
    parser = argparse.ArgumentParser(description='Soybean Meal Option Portfolio RL (Plan B)')
    parser.add_argument('--skip-train', action='store_true',
                        help='Skip training (load saved models)')
    parser.add_argument('--seeds', nargs='+', type=int, default=None,
                        help='Custom random seeds (default: 42 123 456)')
    parser.add_argument('--fetch-only', action='store_true',
                        help='Only fetch data from Tushare, no training')
    args = parser.parse_args()

    seeds = args.seeds if args.seeds else SEEDS

    check_and_make_directories([
        DATA_SAVE_DIR, TRAINED_MODEL_DIR, TENSORBOARD_LOG_DIR, RESULTS_DIR
    ])

    if args.fetch_only:
        from data_loader import fetch_data_from_tushare
        fetch_data_from_tushare()
        print("\n[Done] Data fetched. Set the TUSHARE_TOKEN environment variable first.")
        return

    # Step 1: Data
    hs300_df, hv_df, expire_dates, turnover_dates, base_df = step1_data()

    # Step 2: Features
    train_data, test_data = step2_features(hs300_df, hv_df, expire_dates)

    all_test_metrics = {}
    all_train_metrics = {}

    for seed in seeds:
        print(f"\n{'#' * 60}")
        print(f"# Run with seed={seed}")
        print(f"{'#' * 60}")

        env_train, e_train, e_test = step3_env(train_data, test_data)

        if args.skip_train:
            print(f"\n[Skip train] Loading saved models for seed={seed}...")
            from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3
            loaders = {
                'a2c': A2C.load, 'ddpg': DDPG.load, 'ppo': PPO.load,
                'td3': TD3.load, 'sac': SAC.load,
            }
            models = {}
            for name in ALGORITHMS:
                path = os.path.join(RESULTS_DIR, f'run_seed{seed}', name, f'agent_{name}')
                models[name] = loaders[name](path)
                print(f"  Loaded: {name.upper()}")
        else:
            models = step4_train(env_train, seed=seed)

        test_metrics, train_metrics = step5_evaluate(
            models, e_train, e_test, base_df, seed=seed,
            train_data=train_data,
            test_data=test_data,
            turnover_dates=turnover_dates,
        )

        all_test_metrics[seed] = test_metrics
        all_train_metrics[seed] = train_metrics

    # Step 6: Multi-seed summary
    if len(seeds) > 1:
        step6_summary(all_test_metrics, all_train_metrics, seeds)

    print("\n" + "=" * 60)
    print("Pipeline complete! (Plan B: soybean meal option generalisation evidence)")
    print("=" * 60)


if __name__ == "__main__":
    main()
