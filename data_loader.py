"""
Data loading and contract selection for HS300 ETF option portfolio.

Reads Tushare-format CSV files (option daily bars, contract info, ETF prices),
selects 6 liquid contracts per rebalancing period (3 calls + 3 puts),
and computes historical volatility.
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.stats import norm

from config import (
    OPTION_DAILY_DIR, CONTRACT_FILE, ETF_FILE,
    START_DATE, END_DATE,
)


def load_contracts():
    """Load HS300 ETF option contract info, filter by underlying OP510300.SH."""
    df = pd.read_csv(CONTRACT_FILE, encoding='utf-8-sig')
    df = df[df['opt_code'] == 'OP510300.SH'].copy()
    df = df.rename(columns={
        'ts_code': 'code',
        'call_put': 'contract_type',
        'maturity_date': 'expire_date',
        'opt_code': 'underlying_symbol',
    })
    df['contract_type'] = df['contract_type'].map({'C': 'CO', 'P': 'PO'})
    df['expire_date'] = df['expire_date'].apply(
        lambda x: datetime.strftime(pd.Timestamp(str(int(x))), '%Y-%m-%d')
    )
    return df[['code', 'name', 'contract_type', 'underlying_symbol',
               'exercise_price', 'expire_date']]


def load_options_bar(date1, date2):
    """Load option daily bar data for HS300 ETF options within [date1, date2]."""
    start_dt = pd.Timestamp(date1)
    end_dt = pd.Timestamp(date2)
    all_data = []
    current = start_dt
    while current <= end_dt:
        fname = current.strftime('%Y%m%d') + '.csv'
        fpath = os.path.join(OPTION_DAILY_DIR, fname)
        if os.path.exists(fpath):
            df = pd.read_csv(fpath, encoding='utf-8-sig')
            df = df[df['ts_code'].str.startswith('100') & df['ts_code'].str.endswith('.SH')].copy()
            df = df.rename(columns={
                'ts_code': 'code',
                'trade_date': 'date',
                'amount': 'money',
                'vol': 'volume',
            })
            df['date'] = df['date'].apply(
                lambda x: datetime.strftime(pd.Timestamp(str(int(x))), '%Y-%m-%d')
            )
            all_data.append(df)
        current += pd.Timedelta(days=1)
    if not all_data:
        return pd.DataFrame()
    return pd.concat(all_data, ignore_index=True)


def load_base_etf():
    """Load HS300 ETF (510300.SH) daily closing prices."""
    df = pd.read_csv(ETF_FILE, encoding='utf-8-sig')
    df = df.rename(columns={'trade_date': 'date'})
    df['date'] = df['date'].apply(
        lambda x: datetime.strftime(pd.Timestamp(str(int(x))), '%Y-%m-%d')
    )
    return df[['date', 'close']].sort_values('date').reset_index(drop=True)


def calculate_hv(base_df, window=20):
    """Compute 20-day annualized historical volatility from ETF close prices."""
    df = base_df.copy()
    df['ret'] = df['close'].pct_change()
    df['hv'] = df['ret'].rolling(window=window).std() * np.sqrt(252)
    return df[['date', 'hv']].dropna()


def calculate_bs_greeks(hs300_df):
    """
    Compute Black-Scholes implied volatility and Greeks for each option row.

    Inputs needed (already in hs300_df):
      - close: option market price
      - exercise_price: strike price K
      - base_p: underlying ETF price S
      - diff_days: days to expiration (converted to T in years)
      - contract_type: 'CO' (call) or 'PO' (put)

    Outputs added to hs300_df:
      - iv: implied volatility (solved by bisection)
      - delta: dC/dS (call: N(d1), put: N(d1)-1)
      - gamma: d2C/dS2 (same for call/put)
      - vega: dC/dsigma (same for call/put, scaled by 100)
      - moneyness: S/K
    """
    df = hs300_df.copy()
    r = 0.0

    def _bs_price(S, K, T, sigma, is_call):
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if is_call:
            return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    def _bs_greeks(S, K, T, sigma, is_call):
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        delta = norm.cdf(d1) if is_call else norm.cdf(d1) - 1
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        vega = S * norm.pdf(d1) * np.sqrt(T) / 100
        return delta, gamma, vega

    def _solve_iv(market_price, S, K, T, is_call):
        """Bisection method to find implied volatility."""
        if market_price <= 0 or T <= 0 or S <= 0 or K <= 0:
            return np.nan
        lo, hi = 0.001, 5.0
        for _ in range(100):
            mid = (lo + hi) / 2
            price = _bs_price(S, K, T, mid, is_call)
            if abs(price - market_price) < 1e-6:
                return mid
            if price < market_price:
                lo = mid
            else:
                hi = mid
        return mid

    iv_list, delta_list, gamma_list, vega_list, moneyness_list = [], [], [], [], []
    for _, row in df.iterrows():
        S = float(row['base_p'])
        K = float(row['exercise_price'])
        T = max(float(row['diff_days']) / 252.0, 1e-6)
        is_call = row['contract_type'] == 'CO'
        mp = float(row['close'])

        iv = _solve_iv(mp, S, K, T, is_call)
        if np.isnan(iv):
            delta, gamma, vega = np.nan, np.nan, np.nan
        else:
            delta, gamma, vega = _bs_greeks(S, K, T, iv, is_call)

        iv_list.append(iv)
        delta_list.append(delta)
        gamma_list.append(gamma)
        vega_list.append(vega)
        moneyness_list.append(S / K if K > 0 else np.nan)

    df['iv'] = iv_list
    df['delta'] = delta_list
    df['gamma'] = gamma_list
    df['vega'] = vega_list
    df['moneyness'] = moneyness_list

    for col in ['iv', 'delta', 'gamma', 'vega', 'moneyness']:
        df[col] = df[col].fillna(df[col].mean())

    return df


def calculate_turbulence(data):
    """Calculate turbulence index based on dow 30 approach."""
    df = data.copy()
    df_price_pivot = df.pivot(index="date", columns="tic", values="close")
    df_price_pivot = df_price_pivot.pct_change()

    unique_date = df.date.unique()
    start = 20
    turbulence_index = [0] * start
    count = 0
    for i in range(start, len(unique_date)):
        current_price = df_price_pivot[df_price_pivot.index == unique_date[i]]
        hist_price = df_price_pivot[
            (df_price_pivot.index < unique_date[i])
            & (df_price_pivot.index >= unique_date[i - start])
        ]
        filtered_hist_price = hist_price.iloc[hist_price.isna().sum().min():].dropna(axis=1)

        cov_temp = filtered_hist_price.cov()
        current_temp = current_price[[x for x in filtered_hist_price]] - np.mean(
            filtered_hist_price, axis=0
        )
        temp = current_temp.values.dot(np.linalg.pinv(cov_temp)).dot(current_temp.values.T)
        if temp > 0:
            count += 1
            if count > 2:
                turbulence_temp = temp[0][0]
            else:
                turbulence_temp = 0
        else:
            turbulence_temp = 0
        turbulence_index.append(turbulence_temp)
    try:
        turbulence_index = pd.DataFrame(
            {"date": df_price_pivot.index, "turbulence": turbulence_index}
        )
    except ValueError:
        raise Exception("Turbulence information could not be added.")
    return turbulence_index


def select_contracts(options_bar, options_info, base_df, start_date, end_date):
    """
    Select 6 liquid option contracts per rebalancing period.

    Logic:
      1. Sort by expiration date, determine turnover dates (next trading day after expiry).
      2. Each turnover: pick top-3 calls + top-3 puts by trading value (money).
      3. Assign tic labels: b1/b2/b3 (calls, by strike) + s1/s2/s3 (puts, by strike).
      4. Non-turnover days carry forward the previous period's contracts.
    """
    hs300_options = pd.merge(options_bar, options_info, on='code')
    hs300_options = hs300_options[
        (hs300_options['date'] >= start_date) &
        (hs300_options['date'] <= end_date)
    ].copy().reset_index(drop=True)

    expire_dates = sorted(hs300_options['expire_date'].unique().tolist())
    expire_dates = [d for d in expire_dates if d <= hs300_options['date'].max()]
    trade_dates = sorted(hs300_options['date'].unique().tolist())

    turnover_dates = []
    for ed in expire_dates:
        if ed in trade_dates:
            idx = trade_dates.index(ed)
            if idx + 1 < len(trade_dates):
                turnover_dates.append(trade_dates[idx + 1])

    columns = hs300_options.columns.tolist() + ['tic', 'base_p']
    total_rows = len(trade_dates) * 6
    hs300_df = pd.DataFrame(index=range(total_rows), columns=columns)
    hs300_df['date'] = sorted(trade_dates * 6)
    hs300_df['tic'] = 'bs'
    hs300_df['base_p'] = 0.0

    def _select_6_contracts(tdf, base_val):
        """Pick top-3 calls + top-3 puts by money, sort by strike, assign tic labels."""
        tdf_co = tdf[tdf['contract_type'] == 'CO'].copy()
        tdf_co = tdf_co.loc[tdf_co['money'].nlargest(3).index].sort_values(by='exercise_price')
        tdf_co['tic'] = ['b1', 'b2', 'b3']
        tdf_po = tdf[tdf['contract_type'] == 'PO'].copy()
        tdf_po = tdf_po.loc[tdf_po['money'].nlargest(3).index].sort_values(by='exercise_price')
        tdf_po['tic'] = ['s1', 's2', 's3']
        tdf_sel = pd.concat([tdf_co, tdf_po])
        tdf_sel['base_p'] = base_val
        return tdf_sel

    def _fill_into_df(tdf_sel, hs300_df):
        tdate_val = tdf_sel['date'].iloc[0]
        mask = hs300_df['date'] == tdate_val
        hs300_df.loc[mask] = tdf_sel.values

    # ---- First period ----
    if len(turnover_dates) > 0:
        first_turnover = turnover_dates[0]
        first_expire = expire_dates[0]
        if hs300_options['date'].min() < first_turnover:
            first_date = hs300_options['date'].min()
            tdf = hs300_options[
                (hs300_options['date'] == first_date) &
                (hs300_options['expire_date'] == first_expire)
            ].copy()
        else:
            tdf = hs300_options[
                (hs300_options['date'] == first_turnover) &
                (hs300_options['expire_date'] == first_expire)
            ].copy()
        if len(tdf) >= 6:
            base_val = base_df.loc[base_df['date'] == first_date, 'close'].values[0]
            tdf_sel = _select_6_contracts(tdf, base_val)
            _fill_into_df(tdf_sel, hs300_df)

    # ---- Middle turnover dates ----
    for i in range(1, len(turnover_dates)):
        if i >= len(expire_dates):
            break
        tdate = turnover_dates[i - 1]
        expire_date = expire_dates[i]
        tdf = hs300_options[
            (hs300_options['date'] == tdate) &
            (hs300_options['expire_date'] == expire_date)
        ].copy()
        if len(tdf) < 6:
            continue
        base_val = base_df.loc[base_df['date'] == tdate, 'close'].values[0]
        tdf_sel = _select_6_contracts(tdf, base_val)
        _fill_into_df(tdf_sel, hs300_df)

    # ---- Last turnover date ----
    if len(turnover_dates) > 0:
        last_turnover = turnover_dates[-1]
        future_expires = [d for d in sorted(hs300_options['expire_date'].unique()) if d > last_turnover]
        if future_expires:
            next_expire = future_expires[0]
            tdf = hs300_options[
                (hs300_options['date'] == last_turnover) &
                (hs300_options['expire_date'] == next_expire)
            ].copy()
            if len(tdf) >= 6:
                base_val_arr = base_df.loc[base_df['date'] == last_turnover, 'close'].values
                base_val = base_val_arr[0] if len(base_val_arr) > 0 else 0
                tdf_sel = _select_6_contracts(tdf, base_val)
                _fill_into_df(tdf_sel, hs300_df)

    # ---- Forward fill: non-turnover days carry forward previous contracts ----
    for i in range(len(hs300_df)):
        if pd.isna(hs300_df.loc[i, 'code']):
            prev_code = hs300_df.loc[i - 6, 'code']
            prev_tic = hs300_df.loc[i - 6, 'tic']
            current_date = hs300_df.loc[i, 'date']
            row = hs300_options[
                (hs300_options['date'] == current_date) &
                (hs300_options['code'] == prev_code)
            ]
            if len(row) > 0:
                row = row.iloc[0]
                for col in hs300_options.columns:
                    if col in row.index:
                        hs300_df.loc[i, col] = row[col]
                hs300_df.loc[i, 'tic'] = prev_tic
                base_val_arr = base_df.loc[base_df['date'] == current_date, 'close'].values
                hs300_df.loc[i, 'base_p'] = base_val_arr[0] if len(base_val_arr) > 0 else 0

    hs300_df = hs300_df.dropna(subset=['code']).reset_index(drop=True)
    return hs300_df, expire_dates, turnover_dates


def prepare_data():
    """
    Main data pipeline: load -> select contracts -> compute features -> split.

    Returns:
        train_data, test_data: processed DataFrames ready for the trading env.
        expire_dates, turnover_dates: rebalancing schedule.
        base_df: ETF closing prices for benchmark.
    """
    print("[Data] Loading Tushare option data...")
    base_df = load_base_etf()
    options_bar = load_options_bar(START_DATE, END_DATE)
    options_info = load_contracts()

    print(f"  Option bars: {len(options_bar)} rows, {options_bar['code'].nunique()} contracts")
    print(f"  Contract info: {len(options_info)} rows")
    print(f"  Base ETF: {len(base_df)} trading days")

    hs300_df, expire_dates, turnover_dates = select_contracts(
        options_bar, options_info, base_df, START_DATE, END_DATE
    )
    print(f"  Selected: {len(hs300_df)} rows, {hs300_df['date'].nunique()} days, 6 contracts/day")
    print(f"  Expiry dates: {len(expire_dates)}, Turnover dates: {len(turnover_dates)}")

    # diff_days: days to expiration
    hs300_df['diff_days'] = hs300_df.apply(
        lambda x: (datetime.strptime(str(x.expire_date), '%Y-%m-%d')
                   - datetime.strptime(str(x.date), '%Y-%m-%d')).days + 1,
        axis=1
    )

    # Historical volatility
    hv_df = calculate_hv(base_df)
    print(f"  HV data: {len(hv_df)} rows")
    print("[Data] Done!")
    return hs300_df, hv_df, expire_dates, turnover_dates, base_df
