"""
Backtesting and performance evaluation.

Computes Terminal Value Multiple (TVM), Maximum Drawdown (MDD),
Sharpe Ratio (SR) for each model and the buy-and-hold benchmark.
Also plots cumulative NAV curves.

Supports multi-seed evaluation: results saved to results/run_seed{seed}/.
"""

import os
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from finrl.agents.stablebaselines3.models import DRLAgent


def max_drawdown(returns):
    """Compute maximum drawdown from daily returns series."""
    cumulative = (1 + returns).cumprod()
    peak = cumulative.expanding().max()
    drawdown = (cumulative - peak) / peak
    return drawdown.min()


def sharpe_ratio(returns, risk_free=0, periods=252):
    """Compute annualized Sharpe ratio from daily returns."""
    excess = returns - risk_free
    if excess.std() == 0:
        return 0.0
    return np.sqrt(periods) * excess.mean() / excess.std()


def predict(model, env):
    """Run prediction on a trained model in the given environment."""
    df_account_value, df_actions = DRLAgent.DRL_prediction(
        model=model, environment=env
    )
    return df_account_value, df_actions


def predict_all(models, env):
    """Run prediction for all 5 models. Returns dicts of account values and actions."""
    results = {}
    actions = {}
    for name, model in models.items():
        print(f"[Predict] Running {name.upper()} prediction...")
        df_av, df_act = predict(model, env)
        results[name] = df_av
        actions[name] = df_act
    return results, actions


def compute_holdings_from_actions(df_actions):
    """
    Compute end-of-day holdings from daily trading actions.
    Positive action = buy, negative action = sell.
    Initial holdings = 0. Cumulative sum gives end-of-day holdings.
    Since the environment enforces long-only, all holdings should be >= 0.
    """
    df_hold = df_actions.copy()
    numeric_cols = df_hold.select_dtypes(include='number').columns
    df_hold[numeric_cols] = df_hold[numeric_cols].cumsum()
    return df_hold


def build_benchmark(base_df, start_date, end_date, initial_amount=60000):
    """Build buy-and-hold benchmark: ETF close normalized to initial_amount."""
    df_base = copy.deepcopy(base_df[['date', 'close']])
    df_base = df_base.loc[
        (df_base['date'] >= start_date) & (df_base['date'] <= end_date), :
    ].reset_index(drop=True)
    fst_day = df_base['close'][0]
    benchmark = pd.merge(
        df_base['date'], df_base['close'].div(fst_day).mul(initial_amount),
        how='outer', left_index=True, right_index=True
    ).set_index('date')
    benchmark.columns = ['close']
    return benchmark


def build_equal_weight_benchmark(df_data, turnover_dates, start_date, end_date,
                                  initial_amount=60000, buy_cost=0.0001, sell_cost=0.0001):
    """
    Build equal-weight option portfolio benchmark.

    At each turnover date, allocate capital equally to 6 selected contracts
    (3 calls + 3 puts), hold until next rebalancing, with transaction costs.

    Args:
        df_data: processed DataFrame with daily option data (6 rows per day)
        turnover_dates: list of rebalancing dates (next trading day after expiry)
        start_date: start date for backtest
        end_date: end date for backtest
        initial_amount: initial capital
        buy_cost: buy transaction cost rate
        sell_cost: sell transaction cost rate

    Returns:
        DataFrame with 'account_value' column, indexed by date
    """
    df = df_data.copy()
    df = df[(df['date'] >= start_date) & (df['date'] <= end_date)].reset_index(drop=True)

    trade_dates = sorted(df['date'].unique().tolist())
    stock_dim = df['tic'].nunique()  # should be 6

    # Filter turnover dates within range
    active_turnovers = [d for d in turnover_dates if start_date <= d <= end_date]
    # Ensure first trading day is also a rebalance day (initial allocation)
    if trade_dates and trade_dates[0] not in active_turnovers:
        active_turnovers = [trade_dates[0]] + active_turnovers
    active_turnovers = sorted(set(active_turnovers))

    cash = float(initial_amount)
    holdings = np.zeros(stock_dim)  # number of contracts for each tic
    nav_history = []
    prev_prices = None  # previous trading day's prices (belong to the OLD cycle)

    for i, date in enumerate(trade_dates):
        day_data = df[df['date'] == date].sort_values('tic').reset_index(drop=True)
        prices = day_data['close'].values.astype(float)

        # Rebalance on turnover dates
        if date in active_turnovers:
            # First sell all existing positions.
            # NOTE: ticker slots (b1..b3, s1..s3) are reused across cycles, so on a
            # turnover day `prices` already refer to the NEW contracts. The old
            # holdings must therefore be liquidated at the PREVIOUS trading day's
            # prices, which still correspond to the old contracts -- otherwise the
            # positions are revalued at the wrong contracts' prices and NAV blows up.
            if holdings.sum() > 0:
                sell_prices = prev_prices if prev_prices is not None else prices
                sell_proceeds = np.sum(holdings * sell_prices * (1 - sell_cost))
                cash += sell_proceeds
                holdings = np.zeros(stock_dim)

            # Then buy equal weight
            if cash > 0:
                allocation_per_contract = cash / stock_dim
                for j in range(stock_dim):
                    if prices[j] > 0:
                        # integer number of contracts, constrained by cash
                        max_qty = int(allocation_per_contract / (prices[j] * (1 + buy_cost)))
                        if max_qty > 0:
                            holdings[j] = max_qty
                            cost = holdings[j] * prices[j] * (1 + buy_cost)
                            cash -= cost

        # Calculate end-of-day NAV
        nav = cash + np.sum(holdings * prices)
        nav_history.append({'date': date, 'account_value': nav})
        prev_prices = prices

    result = pd.DataFrame(nav_history).set_index('date')
    return result


def compute_metrics(account_values, benchmark):
    """
    Compute TVM, MDD, Sharpe for all models + benchmark.

    Args:
        account_values: dict {model_name: DataFrame with 'account_value' and 'date'}
        benchmark: DataFrame with 'close' column, indexed by date

    Returns:
        DataFrame with columns: tvm, mdd, sharpe
        DataFrame with all raw values for plotting
    """
    result = pd.DataFrame()
    for name, df_av in account_values.items():
        result[name] = df_av['account_value'].values
        result.index = df_av['date'].values

    result = pd.merge(result, benchmark, how='outer',
                     left_index=True, right_index=True)

    model_names = list(account_values.keys()) + ['close']
    for name in model_names:
        result[f'{name}_r'] = result[name].pct_change()
        first_valid = result[name].dropna().iloc[0]
        result[f'{name}_c'] = result[name] / first_valid

    metrics = []
    for name in model_names:
        tvm = result[f'{name}_c'].dropna().iloc[-1]
        mdd = max_drawdown(result[f'{name}_r'].dropna())
        sr = sharpe_ratio(result[f'{name}_r'].dropna())
        metrics.append({'model': name, 'tvm': tvm, 'mdd': mdd, 'sharpe': sr})

    res_df = pd.DataFrame(metrics)
    res_df = res_df.set_index('model')
    return res_df, result


def plot_nav(result, model_names, title='Portfolio NAV Comparison'):
    """Plot cumulative NAV for all models + benchmark."""
    plt.rcParams['figure.figsize'] = (15, 5)
    fig, ax = plt.subplots()

    ax.plot(result['close'].values, label='Buy & Hold (ETF)', linewidth=1.5)
    markers = {'a2c': '1', 'ddpg': '2', 'ppo': '+', 'td3': '*', 'sac': 'o'}
    for name in model_names:
        if name in result.columns:
            ax.plot(result[name].values, '--', label=name.upper(),
                    linewidth=0.8, marker=markers.get(name, ''))

    ax.legend()
    ax.set_ylabel('NAV')
    ax.set_title(title)
    plt.tight_layout()
    return fig


def save_actions(actions, output_dir):
    """Save trading action logs for each model as CSV."""
    os.makedirs(output_dir, exist_ok=True)
    for name, df_act in actions.items():
        filepath = os.path.join(output_dir, f'{name}.csv')
        df_act.to_csv(filepath, encoding='gbk')
        print(f"[Save] Actions saved: {filepath}")


def save_holdings(holdings, output_dir):
    """Save end-of-day holdings logs for each model as CSV."""
    os.makedirs(output_dir, exist_ok=True)
    for name, df_hold in holdings.items():
        filepath = os.path.join(output_dir, f'{name}_holdings.csv')
        df_hold.to_csv(filepath, encoding='gbk')
        print(f"[Save] Holdings saved: {filepath}")


def evaluate(models, test_env, train_env, base_df,
             test_start, test_end, train_start, train_end,
             initial_amount=60000, seed=42,
             test_data=None, train_data=None, turnover_dates=None):
    """
    Full evaluation pipeline: predict on test set + train set,
    compute metrics, plot NAV curves, save results.

    Results saved to results/run_seed{seed}/.

    Args:
        ... (existing args)
        test_data: processed test DataFrame (for equal-weight benchmark)
        train_data: processed train DataFrame (for equal-weight benchmark)
        turnover_dates: list of rebalancing dates

    Returns:
        test_metrics, train_metrics: DataFrames with TVM/MDD/Sharpe
    """
    run_dir = os.path.join('results', f'run_seed{seed}')
    os.makedirs(run_dir, exist_ok=True)

    # ---- Test set ----
    print("\n" + "=" * 50)
    print(f"[Eval] Test set prediction (seed={seed})...")
    print("=" * 50)
    test_av, test_actions = predict_all(models, test_env)
    test_benchmark = build_benchmark(base_df, test_start, test_end, initial_amount)

    if test_data is not None and turnover_dates is not None:
        ew_test = build_equal_weight_benchmark(
            test_data, turnover_dates, test_start, test_end, initial_amount
        )
        test_av['equal_weight'] = ew_test.reset_index()
        print(f"[Eval] Equal-weight benchmark (test): TVM={ew_test['account_value'].iloc[-1]/initial_amount:.4f}")

    test_metrics, test_result = compute_metrics(test_av, test_benchmark)

    fig_test = plot_nav(test_result, list(models.keys()),
                        title=f'Out-of-Sample NAV Comparison (seed={seed})')
    fig_test.savefig(os.path.join(run_dir, 'nav_test.png'), dpi=150)
    print(f"[Eval] Test plot saved: {run_dir}/nav_test.png")

    # ---- Train set ----
    print("\n" + "=" * 50)
    print(f"[Eval] Train set prediction (seed={seed})...")
    print("=" * 50)
    train_av, train_actions = predict_all(models, train_env)
    train_benchmark = build_benchmark(base_df, train_start, train_end, initial_amount)

    if train_data is not None and turnover_dates is not None:
        ew_train = build_equal_weight_benchmark(
            train_data, turnover_dates, train_start, train_end, initial_amount
        )
        train_av['equal_weight'] = ew_train.reset_index()
        print(f"[Eval] Equal-weight benchmark (train): TVM={ew_train['account_value'].iloc[-1]/initial_amount:.4f}")

    train_metrics, train_result = compute_metrics(train_av, train_benchmark)

    fig_train = plot_nav(train_result, list(models.keys()),
                         title=f'In-Sample NAV Comparison (seed={seed})')
    fig_train.savefig(os.path.join(run_dir, 'nav_train.png'), dpi=150)
    print(f"[Eval] Train plot saved: {run_dir}/nav_train.png")

    # ---- Save actions + holdings ----
    save_actions(test_actions, os.path.join(run_dir, 'weights_test'))
    save_actions(train_actions, os.path.join(run_dir, 'weights_train'))

    test_holdings = {name: compute_holdings_from_actions(df_act) 
                     for name, df_act in test_actions.items()}
    train_holdings = {name: compute_holdings_from_actions(df_act) 
                      for name, df_act in train_actions.items()}
    save_holdings(test_holdings, os.path.join(run_dir, 'weights_test'))
    save_holdings(train_holdings, os.path.join(run_dir, 'weights_train'))

    print("\n" + "=" * 50)
    print(f"[Eval] Out-of-Sample Metrics (seed={seed}):")
    print(test_metrics.to_string())
    print(f"\n[Eval] In-Sample Metrics (seed={seed}):")
    print(train_metrics.to_string())
    print("=" * 50)

    return test_metrics, train_metrics
