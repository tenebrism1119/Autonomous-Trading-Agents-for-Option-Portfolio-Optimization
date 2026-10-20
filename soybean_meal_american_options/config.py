"""
Configuration for Soybean Meal Option Portfolio RL Trading.

Plan B: generalisation evidence - American commodity options
Underlying: soybean meal futures (Dalian Commodity Exchange, DCE)
Options: soybean meal options (American style, CRR binomial tree pricing)
State space: 79 dims = 1(cash) + 6(price) + 6(holdings) + 11x6(indicators)
  New indicators: theta (time decay), early_exercise_premium (early exercise value)
"""

import os

# ---- Tushare API ----
# Never hard-code credentials. Set the token in your environment instead:
#   set TUSHARE_TOKEN=xxxxxxxx   (Windows cmd)   /   export TUSHARE_TOKEN=xxxxxxxx (bash)
TUSHARE_TOKEN = os.environ.get('TUSHARE_TOKEN', '')

# ---- Data Paths ----
DATA_ROOT = os.path.dirname(os.path.abspath(__file__))
LOCAL_DATA_DIR = os.path.join(DATA_ROOT, 'data')
OPTION_DAILY_CSV = os.path.join(LOCAL_DATA_DIR, 'option_daily.csv')
CONTRACT_CSV = os.path.join(LOCAL_DATA_DIR, 'option_contracts.csv')
FUTURES_CSV = os.path.join(LOCAL_DATA_DIR, 'futures_daily.csv')

# ---- Date Range ----
START_DATE = '2022-01-01'
END_DATE = '2023-12-31'
TRAIN_SPLIT_DATE = '2023-04-01'

# ---- Output Directories ----
DATA_SAVE_DIR = 'datasets'
TRAINED_MODEL_DIR = 'trained_models'
TENSORBOARD_LOG_DIR = 'tensorboard_log'
RESULTS_DIR = 'results'

# ---- Contract Parameters ----
CONTRACT_MULTIPLIER = 10
STOCK_DIM = 6  # 3 calls + 3 puts

# state_space = 1 + 2*6 + 11*6 = 79
INDICATORS = [
    'exercise_price', 'base_p', 'diff_days', 'hv',
    'iv', 'delta', 'gamma', 'vega', 'moneyness',
    'theta', 'early_exercise_premium',
]

# ---- Environment Parameters ----
ENV_KWARGS = {
    'hmax': 40,
    'initial_amount': 60000,
    'num_stock_shares': [0] * STOCK_DIM,
    'buy_cost_pct': [0.0001] * STOCK_DIM,
    'sell_cost_pct': [0.0001] * STOCK_DIM,
    'state_space': 1 + 2 * STOCK_DIM + len(INDICATORS) * STOCK_DIM,
    'stock_dim': STOCK_DIM,
    'tech_indicator_list': INDICATORS,
    'action_space': STOCK_DIM,
    'reward_scaling': 1e-1,
    'turbulence_threshold': 90,
}

# ---- CRR Binomial Tree Parameters ----
CRR_STEPS = 100

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

# ---- Algorithms ----
ALGORITHMS = ['a2c', 'ddpg', 'ppo', 'td3', 'sac']

# ---- Multi-Seed ----
SEEDS = [42, 123, 456]
