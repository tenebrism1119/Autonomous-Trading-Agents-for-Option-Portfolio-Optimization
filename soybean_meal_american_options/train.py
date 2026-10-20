"""
Model training for 5 DRL algorithms: A2C, DDPG, PPO, TD3, SAC.

Each function creates and trains a model, saves it to results/run_seed{seed}/{algo}/.
Seed is passed via get_model(seed=...) parameter, NOT model_kwargs.
"""

import os
from finrl.agents.stablebaselines3.models import DRLAgent
from config import (
    TRAIN_TIMESTEPS, PPO_PARAMS, TD3_PARAMS, SAC_PARAMS,
    TRAINED_MODEL_DIR, RESULTS_DIR,
)


def train_a2c(env_train, seed=42):
    """Train A2C agent with random seed for reproducibility."""
    agent = DRLAgent(env=env_train)
    model = agent.get_model('a2c', seed=seed)
    trained = agent.train_model(model=model, tb_log_name=f'a2c_seed{seed}',
                               total_timesteps=TRAIN_TIMESTEPS['a2c'])
    save_path = os.path.join(RESULTS_DIR, f'run_seed{seed}', 'a2c')
    os.makedirs(save_path, exist_ok=True)
    trained.save(os.path.join(save_path, 'agent_a2c'))
    print(f"[Train] A2C done (seed={seed}, {TRAIN_TIMESTEPS['a2c']} steps)")
    return trained


def train_ddpg(env_train, seed=42):
    """Train DDPG agent with random seed for reproducibility."""
    agent = DRLAgent(env=env_train)
    model = agent.get_model('ddpg', seed=seed)
    trained = agent.train_model(model=model, tb_log_name=f'ddpg_seed{seed}',
                               total_timesteps=TRAIN_TIMESTEPS['ddpg'])
    save_path = os.path.join(RESULTS_DIR, f'run_seed{seed}', 'ddpg')
    os.makedirs(save_path, exist_ok=True)
    trained.save(os.path.join(save_path, 'agent_ddpg'))
    print(f"[Train] DDPG done (seed={seed}, {TRAIN_TIMESTEPS['ddpg']} steps)")
    return trained


def train_ppo(env_train, seed=42):
    """Train PPO agent with custom hyperparameters and random seed."""
    agent = DRLAgent(env=env_train)
    model = agent.get_model('ppo', model_kwargs=PPO_PARAMS, seed=seed)
    trained = agent.train_model(model=model, tb_log_name=f'ppo_seed{seed}',
                               total_timesteps=TRAIN_TIMESTEPS['ppo'])
    save_path = os.path.join(RESULTS_DIR, f'run_seed{seed}', 'ppo')
    os.makedirs(save_path, exist_ok=True)
    trained.save(os.path.join(save_path, 'agent_ppo'))
    print(f"[Train] PPO done (seed={seed}, {TRAIN_TIMESTEPS['ppo']} steps)")
    return trained


def train_td3(env_train, seed=42):
    """Train TD3 agent with custom hyperparameters and random seed."""
    agent = DRLAgent(env=env_train)
    model = agent.get_model('td3', model_kwargs=TD3_PARAMS, seed=seed)
    trained = agent.train_model(model=model, tb_log_name=f'td3_seed{seed}',
                               total_timesteps=TRAIN_TIMESTEPS['td3'])
    save_path = os.path.join(RESULTS_DIR, f'run_seed{seed}', 'td3')
    os.makedirs(save_path, exist_ok=True)
    trained.save(os.path.join(save_path, 'agent_td3'))
    print(f"[Train] TD3 done (seed={seed}, {TRAIN_TIMESTEPS['td3']} steps)")
    return trained


def train_sac(env_train, seed=42):
    """Train SAC agent with custom hyperparameters and random seed."""
    agent = DRLAgent(env=env_train)
    model = agent.get_model('sac', model_kwargs=SAC_PARAMS, seed=seed)
    trained = agent.train_model(model=model, tb_log_name=f'sac_seed{seed}',
                               total_timesteps=TRAIN_TIMESTEPS['sac'])
    save_path = os.path.join(RESULTS_DIR, f'run_seed{seed}', 'sac')
    os.makedirs(save_path, exist_ok=True)
    trained.save(os.path.join(save_path, 'agent_sac'))
    print(f"[Train] SAC done (seed={seed}, {TRAIN_TIMESTEPS['sac']} steps)")
    return trained


def train_all(env_train, seed=42):
    """Train all 5 algorithms with given seed. Return dict of trained models."""
    print("\n" + "=" * 50)
    print(f"[Train] Starting training for all 5 algorithms (seed={seed})...")
    print("=" * 50)

    models = {}
    models['a2c'] = train_a2c(env_train, seed=seed)
    models['ddpg'] = train_ddpg(env_train, seed=seed)
    models['ppo'] = train_ppo(env_train, seed=seed)
    models['td3'] = train_td3(env_train, seed=seed)
    models['sac'] = train_sac(env_train, seed=seed)

    print("=" * 50)
    print(f"[Train] All 5 algorithms trained successfully (seed={seed})!")
    print("=" * 50)
    return models
