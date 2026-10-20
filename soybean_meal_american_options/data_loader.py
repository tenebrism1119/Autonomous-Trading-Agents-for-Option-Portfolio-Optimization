"""
Data loading and contract selection for Soybean Meal option portfolio.

Core differences of Plan B:
  1. Read soybean meal option data from local CSV (local option data directory)
  2. Derive futures prices from option put-call parity (no akshare/Tushare needed)
  3. CRR binomial tree pricing (American options, early exercise supported)
  4. Two new state indicators: theta and early_exercise_premium
  5. Contract multiplier = 10 (1 lot = 10 tons)
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy.stats import norm

from config import (
    TUSHARE_TOKEN, LOCAL_DATA_DIR,
    OPTION_DAILY_CSV, CONTRACT_CSV, FUTURES_CSV,
    START_DATE, END_DATE, CONTRACT_MULTIPLIER, CRR_STEPS,
)

OPTION_DATA_ROOT = os.environ.get(
    'OPTION_DATA_ROOT',
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'data', 'options'),
)
LOCAL_CONTRACTS_CSV = os.path.join(OPTION_DATA_ROOT, 'contracts', 'all_option_contracts.csv')
LOCAL_DAILY_DIR = os.path.join(OPTION_DATA_ROOT, 'daily')

FUTURES_CACHE_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'futures_cache.csv')


def load_contracts():
    """Load soybean meal option contracts from the local contract-info CSV."""
    df = pd.read_csv(LOCAL_CONTRACTS_CSV, encoding='utf-8-sig')
    df = df[(df['exchange'] == 'DCE') & (df['ts_code'].str.startswith('M'))].copy()
    df = df.rename(columns={
        'ts_code': 'code',
        'call_put': 'contract_type',
        'maturity_date': 'expire_date',
    })
    df['contract_type'] = df['contract_type'].map({'C': 'CO', 'P': 'PO'})
    df['expire_date'] = df['expire_date'].apply(
        lambda x: datetime.strftime(pd.Timestamp(str(int(x))), '%Y-%m-%d')
        if pd.notna(x) else ''
    )
    cols = ['code', 'name', 'contract_type', 'exercise_price', 'expire_date']
    cols = [c for c in cols if c in df.columns]
    return df[cols].copy()


def load_options_bar():
    """Load soybean meal option daily bars from the local daily CSV files."""
    start_int = int(START_DATE.replace('-', ''))
    end_int = int(END_DATE.replace('-', ''))

    all_dfs = []
    files = sorted(os.listdir(LOCAL_DAILY_DIR))
    for fname in files:
        if not fname.endswith('.csv'):
            continue
        try:
            date_int = int(fname.replace('.csv', ''))
        except ValueError:
            continue
        if date_int < start_int or date_int > end_int:
            continue
        df = pd.read_csv(os.path.join(LOCAL_DAILY_DIR, fname), encoding='utf-8-sig')
        if 'exchange' in df.columns:
            df = df[df['exchange'] == 'DCE']
        if 'ts_code' in df.columns:
            df = df[df['ts_code'].str.startswith('M')]
        if len(df) > 0:
            all_dfs.append(df)

    if not all_dfs:
        raise ValueError(f"No soybean meal option daily data found for {START_DATE}~{END_DATE}")

    df = pd.concat(all_dfs, ignore_index=True)
    df = df.rename(columns={
        'ts_code': 'code',
        'trade_date': 'date',
        'amount': 'money',
        'vol': 'volume',
    })
    df['date'] = df['date'].apply(
        lambda x: datetime.strftime(pd.Timestamp(str(int(x))), '%Y-%m-%d')
    )
    return df


def load_base_futures():
    """Derive soybean meal futures prices from option put-call parity, no external API needed."""
    if os.path.exists(FUTURES_CACHE_CSV):
        print("  [Cache] Loading soybean meal futures daily bars from cache...")
        df = pd.read_csv(FUTURES_CACHE_CSV, encoding='utf-8-sig')
        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
        df = df[(df['date'] >= START_DATE) & (df['date'] <= END_DATE)]
        return df[['date', 'close']].sort_values('date').reset_index(drop=True)

    print("  [Parity] Deriving soybean meal futures prices from option put-call parity...")
    options = load_options_bar()
    contracts = load_contracts()
    merged = pd.merge(options, contracts[['code', 'exercise_price', 'expire_date', 'contract_type']], on='code')

    results = []
    for date, day_df in merged.groupby('date'):
        for expire, expire_df in day_df.groupby('expire_date'):
            calls = expire_df[expire_df['contract_type'] == 'CO']
            puts = expire_df[expire_df['contract_type'] == 'PO']
            merged_cp = pd.merge(
                calls[['exercise_price', 'settle']].rename(columns={'settle': 'call_price'}),
                puts[['exercise_price', 'settle']].rename(columns={'settle': 'put_price'}),
                on='exercise_price'
            )
            if len(merged_cp) == 0:
                continue
            merged_cp['implied_S'] = merged_cp['call_price'] - merged_cp['put_price'] + merged_cp['exercise_price']
            weighted = np.average(merged_cp['implied_S'], weights=merged_cp['call_price'].abs() + merged_cp['put_price'].abs())
            results.append({'date': date, 'close': weighted})

    df = pd.DataFrame(results)
    df = df.groupby('date')['close'].mean().reset_index()
    df = df.sort_values('date').reset_index(drop=True)

    df.to_csv(FUTURES_CACHE_CSV, index=False, encoding='utf-8-sig')
    print(f"  [Cache] Futures data cached: {FUTURES_CACHE_CSV}")
    return df


def calculate_hv(base_df, window=20):
    """20-day annualised historical volatility."""
    df = base_df.copy()
    df['ret'] = df['close'].pct_change()
    df['hv'] = df['ret'].rolling(window=window).std() * np.sqrt(252)
    return df[['date', 'hv']].dropna()


def crr_american_option(S, K, T, r, sigma, is_call, n_steps=100):
    """CRR binomial tree pricing of American options."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        if is_call:
            return max(S - K, 0)
        else:
            return max(K - S, 0)

    dt = T / n_steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp(r * dt) - d) / (u - d)
    disc = np.exp(-r * dt)

    j = np.arange(n_steps + 1)
    stock_prices = S * u**j * d**(n_steps - j)

    if is_call:
        option_values = np.maximum(stock_prices - K, 0)
    else:
        option_values = np.maximum(K - stock_prices, 0)

    for i in range(n_steps - 1, -1, -1):
        j = np.arange(i + 1)
        stock_prices = S * u**j * d**(i - j)
        hold = disc * (p * option_values[1:i + 2] + (1 - p) * option_values[0:i + 1])
        if is_call:
            exercise = np.maximum(stock_prices - K, 0)
        else:
            exercise = np.maximum(K - stock_prices, 0)
        option_values = np.maximum(hold, exercise)

    return float(option_values[0])


def bs_european_option(S, K, T, r, sigma, is_call):
    """Black-Scholes European option pricing (used to compute the early exercise premium)."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return max((S - K) if is_call else (K - S), 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if is_call:
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def solve_iv_crr(market_price, S, K, T, is_call, n_steps=100):
    """Solve implied volatility by bisection (using CRR pricing)."""
    if market_price <= 0 or T <= 0 or S <= 0 or K <= 0:
        return np.nan
    lo, hi = 0.001, 5.0
    mid = 0.3
    for _ in range(50):
        mid = (lo + hi) / 2
        price = crr_american_option(S, K, T, 0.0, mid, is_call, n_steps)
        if abs(price - market_price) < 1e-4:
            return mid
        if price < market_price:
            lo = mid
        else:
            hi = mid
    return mid


def calculate_crr_greeks(hs300_df):
    """Compute Greeks and features of American options with a CRR binomial tree."""
    df = hs300_df.copy()
    r = 0.0
    n = CRR_STEPS

    iv_list, delta_list, gamma_list, vega_list = [], [], [], []
    theta_list, eep_list, moneyness_list = [], [], []

    total = len(df)
    for idx, (_, row) in enumerate(df.iterrows()):
        S = float(row['base_p'])
        K = float(row['exercise_price'])
        T = max(float(row['diff_days']) / 252.0, 1e-6)
        is_call = row['contract_type'] == 'CO'
        mp = float(row['close'])

        iv = solve_iv_crr(mp, S, K, T, is_call, n)

        if np.isnan(iv) or iv <= 0:
            iv, delta, gamma, vega, theta, eep = [np.nan] * 6
        else:
            h_s = max(S * 0.01, 0.01)
            c_up = crr_american_option(S + h_s, K, T, r, iv, is_call, n)
            c_dn = crr_american_option(S - h_s, K, T, r, iv, is_call, n)
            delta = (c_up - c_dn) / (2 * h_s)

            c_mid = crr_american_option(S, K, T, r, iv, is_call, n)
            gamma = (c_up - 2 * c_mid + c_dn) / (h_s ** 2)

            h_v = max(iv * 0.01, 0.001)
            c_vup = crr_american_option(S, K, T, r, iv + h_v, is_call, n)
            c_vdn = crr_american_option(S, K, T, r, iv - h_v, is_call, n)
            vega = (c_vup - c_vdn) / (2 * h_v)

            h_t = 1.0 / 252.0
            c_t_dn = crr_american_option(S, K, max(T - h_t, 1e-6), r, iv, is_call, n)
            c_t_up = crr_american_option(S, K, T + h_t, r, iv, is_call, n)
            theta = -(c_t_up - c_t_dn) / (2 * h_t)

            american_price = c_mid
            european_price = bs_european_option(S, K, T, r, iv, is_call)
            eep = american_price - european_price

        iv_list.append(iv)
        delta_list.append(delta)
        gamma_list.append(gamma)
        vega_list.append(vega)
        theta_list.append(theta)
        eep_list.append(eep)
        moneyness_list.append(S / K if K > 0 else np.nan)

        if (idx + 1) % 500 == 0:
            print(f"  CRR Greeks: {idx + 1}/{total} rows done")

    df['iv'] = iv_list
    df['delta'] = delta_list
    df['gamma'] = gamma_list
    df['vega'] = vega_list
    df['theta'] = theta_list
    df['early_exercise_premium'] = eep_list
    df['moneyness'] = moneyness_list

    for col in ['iv', 'delta', 'gamma', 'vega', 'theta', 'early_exercise_premium', 'moneyness']:
        df[col] = df[col].fillna(df[col].mean())

    print(f"  CRR Greeks: {total}/{total} rows done")
    return df


def calculate_turbulence(data):
    """Compute the turbulence index."""
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
        if filtered_hist_price.empty or len(filtered_hist_price.columns) < 2:
            turbulence_index.append(0)
            continue
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

    turbulence_index = pd.DataFrame(
        {"date": df_price_pivot.index, "turbulence": turbulence_index}
    )
    return turbulence_index


def select_contracts(options_bar, options_info, base_df, start_date, end_date):
    """Select the 6 most liquid contracts (3 calls + 3 puts)."""
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
    """Main data pipeline: local CSV + put-call parity futures -> contract selection -> features."""
    print("[Data] Loading soybean meal option data from local files...")
    base_df = load_base_futures()
    options_bar = load_options_bar()
    options_info = load_contracts()

    print(f"  Option daily bars: {len(options_bar)} rows, {options_bar['code'].nunique()} contracts")
    print(f"  Contract info: {len(options_info)} rows")
    print(f"  Futures daily bars: {len(base_df)} trading days")

    hs300_df, expire_dates, turnover_dates = select_contracts(
        options_bar, options_info, base_df, START_DATE, END_DATE
    )
    print(f"  After selection: {len(hs300_df)} rows, {hs300_df['date'].nunique()} days, 6 contracts/day")
    print(f"  Expiry dates: {len(expire_dates)}, roll dates: {len(turnover_dates)}")

    hs300_df['diff_days'] = hs300_df.apply(
        lambda x: (datetime.strptime(str(x.expire_date), '%Y-%m-%d')
                   - datetime.strptime(str(x.date), '%Y-%m-%d')).days + 1,
        axis=1
    )

    hv_df = calculate_hv(base_df)
    print(f"  HV data: {len(hv_df)} rows")
    print("[Data] Data loading complete!")
    return hs300_df, hv_df, expire_dates, turnover_dates, base_df


def fetch_data_from_tushare():
    """Compatibility wrapper: local data already exists, no fetch from Tushare needed."""
    print("[Data] Local option data already exists, no API fetch needed!")
    print(f"  Data path: {OPTION_DATA_ROOT}")
    print("  Futures data: derived from option put-call parity")
    prepare_data()
