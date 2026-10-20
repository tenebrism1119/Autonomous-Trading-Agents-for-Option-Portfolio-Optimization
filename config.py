"""
Configuration for HS300 ETF Option Portfolio RL Trading.

All paths, date ranges, hyperparameters, and indicator lists live here.
"""

import os

# ---- Data Paths ----
# Override with the DATA_ROOT environment variable if the data lives elsewhere.
DATA_ROOT = os.environ.get(
    'DATA_ROOT',
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'data'),
)
OPTION_DAILY_DIR = os.path.join(DATA_ROOT, 'options', 'daily')
CONTRACT_FILE = os.path.join(DATA_ROOT, 'options', 'contracts', 'all_option_contracts.csv')
ETF_FILE = os.path.join(DATA_ROOT, 'underlyings', 'etf', '510300.SH.csv')

# ---- Date Range ----
START_DATE = '2022-01-01'
END_DATE = '2023-12-31'

# Train/test split of the extended-sample experiment (no validation set).
TRAIN_SPLIT_DATE = '2023-04-01'

# ---- Output Directories ----
DATA_SAVE_DIR = 'datasets'
TRAINED_MODEL_DIR = 'trained_models'
TENSORBOARD_LOG_DIR = 'tensorboard_log'
RESULTS_DIR = 'results'

INDICATORS = ['exercise_price', 'base_p', 'diff_days', 'hv',
              'iv', 'delta', 'gamma', 'vega', 'moneyness']

# ---- Environment Parameters ----
ENV_KWARGS = {
    'hmax': 30000,
    'initial_amount': 60000,
    'num_stock_shares': [0] * 6,
    'buy_cost_pct': [0.0001] * 6,
    'sell_cost_pct': [0.0001] * 6,
    'state_space': 1 + 2 * 6 + len(INDICATORS) * 6,
    'stock_dim': 6,
    'tech_indicator_list': INDICATORS,
    'action_space': 6,
    'reward_scaling': 1e-1,
    'turbulence_threshold': 90,
}

# ---- Training Hyperparameters ----
TRAIN_TIMESTEPS = {
    'a2c': 50000,
    'ddpg': 50000,
    'ppo': 200000,
    'td3': 50000,
    'sac': 70000,
}

PPO_PARAMS = {
    'n_steps': 2048,
    'ent_coef': 0.01,
    'learning_rate': 0.00025,
    'batch_size': 128,
}

TD3_PARAMS = {
    'batch_size': 50,
    'buffer_size': 100000,
    'learning_rate': 0.001,
}

SAC_PARAMS = {
    'batch_size': 50,
    'buffer_size': 100000,
    'learning_rate': 0.0001,
    'learning_starts': 100,
    'ent_coef': 'auto_0.1',
}

# ---- Algorithms to Train ----
ALGORITHMS = ['a2c', 'ddpg', 'ppo', 'td3', 'sac']

SEEDS = [42, 123, 456]
