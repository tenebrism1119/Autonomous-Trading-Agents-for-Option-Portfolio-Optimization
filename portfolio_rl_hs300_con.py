#%%
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime

from finrl.agents. stablebaselines3.models import DRLAgent
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.meta.data_processor import DataProcessor
from stable_baselines3.common.logger import configure

from finrl.plot import backtest_stats, backtest_plot, get_daily_return, get_baseline
from pprint import pprint

# from finrl.config import config
# from finrl.preprocessing.preprocessors import FeatureEngineer

from gymnasium import spaces
from gymnasium.utils import seeding
import gymnasium as gym
from gymnasium.utils import seeding

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

import sys
from finrl import config

import pymysql
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

#%%
def calculate_turbulence(data):
    """calculate turbulence index based on dow 30"""
    # can add other market assets
    df = data.copy()
    df_price_pivot = df.pivot(index="date", columns="tic", values="close")
    # use returns to calculate turbulence
    df_price_pivot = df_price_pivot.pct_change()

    unique_date = df.date.unique()
    # start after a year
#     start = 252
    start = 20
    turbulence_index = [0] * start
    # turbulence_index = [0]
    count = 0
    for i in range(start, len(unique_date)):
        current_price = df_price_pivot[df_price_pivot.index == unique_date[i]]
        # use one year rolling window to calcualte covariance
        hist_price = df_price_pivot[
            (df_price_pivot.index < unique_date[i])
            & (df_price_pivot.index >= unique_date[i - start])
        ]
        # Drop tickers which has number missing values more than the "oldest" ticker
        filtered_hist_price = hist_price.iloc[
            hist_price.isna().sum().min() :
        ].dropna(axis=1)

        cov_temp = filtered_hist_price.cov()
        current_temp = current_price[[x for x in filtered_hist_price]] - np.mean(
            filtered_hist_price, axis=0
        )
        # cov_temp = hist_price.cov()
        # current_temp=(current_price - np.mean(hist_price,axis=0))

        temp = current_temp.values.dot(np.linalg.pinv(cov_temp)).dot(
            current_temp.values.T
        )
        if temp > 0:
            count += 1
            if count > 2:
                turbulence_temp = temp[0][0]
            else:
                # avoid large outlier because of the calculation just begins
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

class StockTradingEnv(gym.Env):
    """A stock trading environment for OpenAI gym"""

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        stock_dim: int,
        hmax: int,
        initial_amount: int,
        num_stock_shares: list[int],
        buy_cost_pct: list[float],  
        sell_cost_pct: list[float],
        reward_scaling: float,
        state_space: int,
        action_space: int,
        tech_indicator_list: list[str],
        turbulence_threshold=None,
        risk_indicator_col="turbulence",
        make_plots: bool = False,
        print_verbosity=10,
        day=0,
        initial=True,
        previous_state=[],
        model_name="",
        mode="",
        iteration="",
    ):
        self.day = day
        self.df = df
        self.stock_dim = stock_dim
        self.hmax = hmax
        self.num_stock_shares = num_stock_shares
        self.initial_amount = initial_amount  # get the initial cash
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.reward_scaling = reward_scaling
        self.state_space = state_space
        self.action_space = action_space
        self.tech_indicator_list = tech_indicator_list
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.action_space,))
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_space,)
        )
        self.data = self.df.loc[self.day, :]
        self.terminal = False
        self.make_plots = make_plots
        self.print_verbosity = print_verbosity
        self.turbulence_threshold = turbulence_threshold
        self.risk_indicator_col = risk_indicator_col
        self.initial = initial
        self.previous_state = previous_state
        self.model_name = model_name
        self.mode = mode
        self.iteration = iteration
        # initalize state
        self.state = self._initiate_state()

        # initialize reward
        self.reward = 0
        self.turbulence = 0
        self.cost = 0
        self.trades = 0
        self.episode = 0
        # memorize all the total balance change
        self.asset_memory = [
            self.initial_amount
            + np.sum(
                np.array(self.num_stock_shares)
                * np.array(self.state[1 : 1 + self.stock_dim])
            )
        ]  # the initial total asset is calculated by cash + sum (num_share_stock_i * price_stock_i)
        self.rewards_memory = []
        self.actions_memory = []
        self.state_memory = (
            []
        )  # we need sometimes to preserve the state in the middle of trading process
        self.date_memory = [self._get_date()]
        #         self.logger = Logger('results',[CSVOutputFormat])
        # self.reset()
        self._seed()

    def _sell_stock(self, index, action):
        def _do_sell_normal():
            if (
                self.state[index + 2 * self.stock_dim + 1] != True
            ):  # check if the stock is able to sell, for simlicity we just add it in techical index
                # if self.state[index + 1] > 0: # if we use price<0 to denote a stock is unable to trade in that day, the total asset calculation may be wrong for the price is unreasonable
                # Sell only if the price is > 0 (no missing data in this particular date)
                # perform sell action based on the sign of the action
                if self.state[index + self.stock_dim + 1] > 0:
                    # Sell only if current asset is > 0
                    sell_num_shares = min(
                        abs(action), self.state[index + self.stock_dim + 1]
                    )
                    sell_amount = (
                        self.state[index + 1]
                        * sell_num_shares
                        * (1 - self.sell_cost_pct[index])
                    )
                    # update balance
                    self.state[0] += sell_amount

                    self.state[index + self.stock_dim + 1] -= sell_num_shares
                    self.cost += (
                        self.state[index + 1]
                        * sell_num_shares
                        * self.sell_cost_pct[index]
                    )
                    self.trades += 1
                else:
                    sell_num_shares = 0
            else:
                sell_num_shares = 0

            return sell_num_shares

        # perform sell action based on the sign of the action
        if self.turbulence_threshold is not None:
            if self.turbulence >= self.turbulence_threshold:
                if self.state[index + 1] > 0:
                    # Sell only if the price is > 0 (no missing data in this particular date)
                    # if turbulence goes over threshold, just clear out all positions
                    if self.state[index + self.stock_dim + 1] > 0:
                        # Sell only if current asset is > 0
                        sell_num_shares = self.state[index + self.stock_dim + 1]
                        sell_amount = (
                            self.state[index + 1]
                            * sell_num_shares
                            * (1 - self.sell_cost_pct[index])
                        )
                        # update balance
                        self.state[0] += sell_amount
                        self.state[index + self.stock_dim + 1] = 0
                        self.cost += (
                            self.state[index + 1]
                            * sell_num_shares
                            * self.sell_cost_pct[index]
                        )
                        self.trades += 1
                    else:
                        sell_num_shares = 0
                else:
                    sell_num_shares = 0
            else:
                sell_num_shares = _do_sell_normal()
        else:
            sell_num_shares = _do_sell_normal()

        return sell_num_shares

    def _buy_stock(self, index, action):
        def _do_buy():
            if (
                self.state[index + 2 * self.stock_dim + 1] != True
            ):  # check if the stock is able to buy
                # if self.state[index + 1] >0:
                # Buy only if the price is > 0 (no missing data in this particular date)
                available_amount = self.state[0] // (
                    self.state[index + 1] * (1 + self.buy_cost_pct[index])
                )  # when buying stocks, we should consider the cost of trading when calculating available_amount, or we may be have cash<0
                # print('available_amount:{}'.format(available_amount))

                # update balance
                buy_num_shares = min(available_amount, action)
                buy_amount = (
                    self.state[index + 1]
                    * buy_num_shares
                    * (1 + self.buy_cost_pct[index])
                )
                self.state[0] -= buy_amount

                self.state[index + self.stock_dim + 1] += buy_num_shares

                self.cost += (
                    self.state[index + 1] * buy_num_shares * self.buy_cost_pct[index]
                )
                self.trades += 1
            else:
                buy_num_shares = 0

            return buy_num_shares

        # perform buy action based on the sign of the action
        if self.turbulence_threshold is None:
            buy_num_shares = _do_buy()
        else:
            if self.turbulence < self.turbulence_threshold:
                buy_num_shares = _do_buy()
            else:
                buy_num_shares = 0
                pass

        return buy_num_shares


    def _make_plot(self):
        plt.plot(self.asset_memory, "r")
        plt.savefig(f"results/account_value_trade_{self.episode}.png")
        plt.close()

    def step(self, actions):
        self.terminal = self.day >= len(self.df.index.unique()) - 1
        if self.terminal:
            # print(f"Episode: {self.episode}")
            if self.make_plots:
                self._make_plot()
            end_total_asset = self.state[0] + sum(
                np.array(self.state[1 : (self.stock_dim + 1)])
                * np.array(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
            )
            df_total_value = pd.DataFrame(self.asset_memory)
            tot_reward = (
                self.state[0]
                + sum(
                    np.array(self.state[1 : (self.stock_dim + 1)])
                    * np.array(
                        self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)]
                    )
                )
                - self.asset_memory[0]
            )  # initial_amount is only cash part of our initial asset
            df_total_value.columns = ["account_value"]
            df_total_value["date"] = self.date_memory
            df_total_value["daily_return"] = df_total_value["account_value"].pct_change(
                1
            )
            if df_total_value["daily_return"].std() != 0:
                sharpe = (
                    (252**0.5)
                    * df_total_value["daily_return"].mean()
                    / df_total_value["daily_return"].std()
                )
            df_rewards = pd.DataFrame(self.rewards_memory)
            df_rewards.columns = ["account_rewards"]
            df_rewards["date"] = self.date_memory[:-1]
            if self.episode % self.print_verbosity == 0:
                print(f"day: {self.day}, episode: {self.episode}")
                print(f"begin_total_asset: {self.asset_memory[0]:0.2f}")
                print(f"end_total_asset: {end_total_asset:0.2f}")
                print(f"total_reward: {tot_reward:0.2f}")
                print(f"total_cost: {self.cost:0.2f}")
                print(f"total_trades: {self.trades}")
                if df_total_value["daily_return"].std() != 0:
                    print(f"Sharpe: {sharpe:0.3f}")
                print("=================================")

            if (self.model_name != "") and (self.mode != ""):
                df_actions = self.save_action_memory()
                df_actions.to_csv(
                    "results/actions_{}_{}_{}.csv".format(
                        self.mode, self.model_name, self.iteration
                    )
                )
                df_total_value.to_csv(
                    "results/account_value_{}_{}_{}.csv".format(
                        self.mode, self.model_name, self.iteration
                    ),
                    index=False,
                )
                df_rewards.to_csv(
                    "results/account_rewards_{}_{}_{}.csv".format(
                        self.mode, self.model_name, self.iteration
                    ),
                    index=False,
                )
                plt.plot(self.asset_memory, "r")
                plt.savefig(
                    "results/account_value_{}_{}_{}.png".format(
                        self.mode, self.model_name, self.iteration
                    )
                )
                plt.close()

            # Add outputs to logger interface
            # logger.record("environment/portfolio_value", end_total_asset)
            # logger.record("environment/total_reward", tot_reward)
            # logger.record("environment/total_reward_pct", (tot_reward / (end_total_asset - tot_reward)) * 100)
            # logger.record("environment/total_cost", self.cost)
            # logger.record("environment/total_trades", self.trades)

            return self.state, self.reward, self.terminal, False, {}

        else:
            actions = actions * self.hmax  # actions initially is scaled between 0 to 1
            actions = actions.astype(
                int
            )  # convert into integer because we can't by fraction of shares
            if self.turbulence_threshold is not None:
                if self.turbulence >= self.turbulence_threshold:
                    actions = np.array([-self.hmax] * self.stock_dim)
        
            begin_total_asset = self.state[0] + sum(
                np.array(self.state[1 : (self.stock_dim + 1)])
                * np.array(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
            )
            # print("begin_total_asset:{}".format(begin_total_asset))

            argsort_actions = np.argsort(actions)
            sell_index = argsort_actions[: np.where(actions < 0)[0].shape[0]]
            buy_index = argsort_actions[::-1][: np.where(actions > 0)[0].shape[0]]

            for index in sell_index:
                # print(f"Num shares before: {self.state[index+self.stock_dim+1]}")
                # print(f'take sell action before : {actions[index]}')
                actions[index] = self._sell_stock(index, actions[index]) * (-1)
                # print(f'take sell action after : {actions[index]}')
                # print(f"Num shares after: {self.state[index+self.stock_dim+1]}")

            for index in buy_index:
                # print('take buy action: {}'.format(actions[index]))
                actions[index] = self._buy_stock(index, actions[index])

            self.actions_memory.append(actions)

            # state: s -> s+1
            self.day += 1
            self.data = self.df.loc[self.day, :]
            if self.turbulence_threshold is not None:
                if len(self.df.tic.unique()) == 1:
                    self.turbulence = self.data[self.risk_indicator_col]
                elif len(self.df.tic.unique()) > 1:
                    self.turbulence = self.data[self.risk_indicator_col].values[0]
            self.state = self._update_state()

            end_total_asset = self.state[0] + sum(
                np.array(self.state[1 : (self.stock_dim + 1)])
                * np.array(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
            )
            self.asset_memory.append(end_total_asset)
            self.date_memory.append(self._get_date())
            self.reward = end_total_asset - begin_total_asset
            self.rewards_memory.append(self.reward)
            self.reward = self.reward * self.reward_scaling
            self.state_memory.append(
                self.state
            )  # add current state in state_recorder for each step

        return self.state, self.reward, self.terminal, False, {}

    def reset(
        self,
        *,
        seed=None,
        options=None,
    ):
        # initiate state
        self.day = 0
        self.data = self.df.loc[self.day, :]
        self.state = self._initiate_state()

        if self.initial:
            self.asset_memory = [
                self.initial_amount
                + np.sum(
                    np.array(self.num_stock_shares)
                    * np.array(self.state[1 : 1 + self.stock_dim])
                )
            ]
        else:
            previous_total_asset = self.previous_state[0] + sum(
                np.array(self.state[1 : (self.stock_dim + 1)])
                * np.array(
                    self.previous_state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)]
                )
            )
            self.asset_memory = [previous_total_asset]

        self.turbulence = 0
        self.cost = 0
        self.trades = 0
        self.terminal = False
        # self.iteration=self.iteration
        self.rewards_memory = []
        self.actions_memory = []
        self.date_memory = [self._get_date()]

        self.episode += 1

        return self.state, {}

    def render(self, mode="human", close=False):
        return self.state

    def _initiate_state(self):
        if self.initial:
            # For Initial State
            if len(self.df.tic.unique()) > 1:
                # for multiple stock
                state = (
                    [self.initial_amount]
                    + self.data.close.values.tolist()
                    + self.num_stock_shares
                    + sum(
                        (
                            self.data[tech].values.tolist()
                            for tech in self.tech_indicator_list
                        ),
                        [],
                    )
                )  # append initial stocks_share to initial state, instead of all zero
            else:
                # for single stock
                state = (
                    [self.initial_amount]
                    + [self.data.close]
                    + [0] * self.stock_dim
                    + sum(([self.data[tech]] for tech in self.tech_indicator_list), [])
                )
        else:
            # Using Previous State
            if len(self.df.tic.unique()) > 1:
                # for multiple stock
                state = (
                    [self.previous_state[0]]
                    + self.data.close.values.tolist()
                    + self.previous_state[
                        (self.stock_dim + 1) : (self.stock_dim * 2 + 1)
                    ]
                    + sum(
                        (
                            self.data[tech].values.tolist()
                            for tech in self.tech_indicator_list
                        ),
                        [],
                    )
                )
            else:
                # for single stock
                state = (
                    [self.previous_state[0]]
                    + [self.data.close]
                    + self.previous_state[
                        (self.stock_dim + 1) : (self.stock_dim * 2 + 1)
                    ]
                    + sum(([self.data[tech]] for tech in self.tech_indicator_list), [])
                )
        return state

    def _update_state(self):
        if len(self.df.tic.unique()) > 1:
            # for multiple stock
            state = (
                [self.state[0]]
                + self.data.close.values.tolist()
                + list(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
                + sum(
                    (
                        self.data[tech].values.tolist()
                        for tech in self.tech_indicator_list
                    ),
                    [],
                )
            )

        else:
            # for single stock
            state = (
                [self.state[0]]
                + [self.data.close]
                + list(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
                + sum(([self.data[tech]] for tech in self.tech_indicator_list), [])
            )

        return state

    def _get_date(self):
        if len(self.df.tic.unique()) > 1:
            date = self.data.date.unique()[0]
        else:
            date = self.data.date
        return date

    # add save_state_memory to preserve state in the trading process
    def save_state_memory(self):
        if len(self.df.tic.unique()) > 1:
            # date and close price length must match actions length
            date_list = self.date_memory[:-1]
            df_date = pd.DataFrame(date_list)
            df_date.columns = ["date"]

            state_list = self.state_memory
            df_states = pd.DataFrame(
                state_list,
                columns=[
                    "cash",
                    "Bitcoin_price",
                    "Gold_price",
                    "Bitcoin_num",
                    "Gold_num",
                    "Bitcoin_Disable",
                    "Gold_Disable",
                ],
            )
            df_states.index = df_date.date
            # df_actions = pd.DataFrame({'date':date_list,'actions':action_list})
        else:
            date_list = self.date_memory[:-1]
            state_list = self.state_memory
            df_states = pd.DataFrame({"date": date_list, "states": state_list})
        # print(df_states)
        return df_states

    def save_asset_memory(self):
        date_list = self.date_memory
        asset_list = self.asset_memory
        # print(len(date_list))
        # print(len(asset_list))
        df_account_value = pd.DataFrame(
            {"date": date_list, "account_value": asset_list}
        )
        return df_account_value

    def save_action_memory(self):
        if len(self.df.tic.unique()) > 1:
            # date and close price length must match actions length
            date_list = self.date_memory[:-1]
            df_date = pd.DataFrame(date_list)
            df_date.columns = ["date"]

            action_list = self.actions_memory
            df_actions = pd.DataFrame(action_list)
            df_actions.columns = self.data.tic.values
            df_actions.index = df_date.date
            # df_actions = pd.DataFrame({'date':date_list,'actions':action_list})
        else:
            date_list = self.date_memory[:-1]
            action_list = self.actions_memory
            df_actions = pd.DataFrame({"date": date_list, "actions": action_list})
        return df_actions

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_sb_env(self):
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs

    
class OptionsPortfolioEnv(gym.Env):
    """A single stock trading environment for OpenAI gym

    Attributes
    ----------
        df: DataFrame
            input data
        stock_dim : int
            number of unique stocks
        hmax : int
            maximum number of shares to trade
        initial_amount : int
            start money
        transaction_cost_pct: float
            transaction cost percentage per trade
        reward_scaling: float
            scaling factor for reward, good for training
        state_space: int
            the dimension of input features
        action_space: int
            equals stock dimension
        tech_indicator_list: list
            a list of technical indicator names
        turbulence_threshold: int
            a threshold to control risk aversion
        day: int
            an increment number to control date

    Methods
    -------
    _sell_stock()
        perform sell action based on the sign of the action
    _buy_stock()
        perform buy action based on the sign of the action
    step()
        at each step the agent will return actions, then
        we will calculate the reward, and return the next observation.
    reset()
        reset the environment
    render()
        use render to return other functions
    save_asset_memory()
        return account value at each time step
    save_action_memory()
        return actions/positions at each time step
        
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        df,
        stock_dim,
        hmax,
        initial_amount,
        transaction_cost_pct,
        reward_scaling,
        state_space,
        action_space,
        tech_indicator_list,
        turbulence_threshold=None,
        lookback=252,
        day=0,
    ):
        # super(StockEnv, self).__init__()
        # money = 10 , scope = 1
        self.day = day
        self.lookback = lookback
        self.df = df
        self.stock_dim = stock_dim
        self.hmax = hmax
        self.initial_amount = initial_amount
        self.transaction_cost_pct = transaction_cost_pct
        self.reward_scaling = reward_scaling
        self.state_space = state_space
        self.action_space = action_space
        self.tech_indicator_list = tech_indicator_list

        # action_space normalization and shape is self.stock_dim
        self.action_space = spaces.Box(low=0, high=1, shape=(self.action_space,))
        # Shape = (34, 30)
        # covariance matrix + technical indicators
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.state_space + len(self.tech_indicator_list), self.state_space),
        )

        # load data from a pandas dataframe
        self.data = self.df.loc[self.day, :]
        self.covs = self.data["cov_list"].values[0]
        self.state = np.append(
            [self.covs],
            [self.data[tech].values.tolist() for tech in self.tech_indicator_list][0],
            axis=0,
        )
        self.terminal = False
        self.turbulence_threshold = turbulence_threshold
        # initalize state: inital portfolio return + individual stock return + individual weights
        self.portfolio_value = self.initial_amount

        # memorize portfolio value each step
        self.asset_memory = [self.initial_amount]
        # memorize portfolio return each step
        self.portfolio_return_memory = [0]
        self.actions_memory = [[1 / self.stock_dim] * self.stock_dim]
        self.date_memory = [self.data.date.unique()[0]]

    def step(self, actions):
        # print(self.day)
        self.terminal = self.day >= len(self.df.index.unique()) - 1
        # print(actions)

        if self.terminal:
            df = pd.DataFrame(self.portfolio_return_memory)
            df.columns = ["daily_return"]
            plt.plot(df.daily_return.cumsum(), "r")
            plt.savefig("results/cumulative_reward.png")
            plt.close()

            plt.plot(self.portfolio_return_memory, "r")
            plt.savefig("results/rewards.png")
            plt.close()

            print("=================================")
            print(f"begin_total_asset:{self.asset_memory[0]}")
            print(f"end_total_asset:{self.portfolio_value}")

            df_daily_return = pd.DataFrame(self.portfolio_return_memory)
            df_daily_return.columns = ["daily_return"]
            if df_daily_return["daily_return"].std() != 0:
                sharpe = (
                    (252**0.5)
                    * df_daily_return["daily_return"].mean()
                    / df_daily_return["daily_return"].std()
                )
                print("Sharpe: ", sharpe)
            print("=================================")

            return self.state, self.reward, self.terminal, False, {}

        else:
            # print("Model actions: ",actions)
            # actions are the portfolio weight
            # normalize to sum of 1
            # if (np.array(actions) - np.array(actions).min()).sum() != 0:
            #  norm_actions = (np.array(actions) - np.array(actions).min()) /
            #                   (np.array(actions) - np.array(actions).min()).sum()
            # else:
            #  norm_actions = actions
            weights = self.softmax_normalization(actions)
            # print("Normalized actions: ", weights)
            self.actions_memory.append(weights)
            last_day_memory = self.data

            # load next state
            self.day += 1
            self.data = self.df.loc[self.day, :]
            self.covs = self.data["cov_list"].values[0]
            self.state = np.append(
                [self.covs],
                [self.data[tech].values.tolist() for tech in self.tech_indicator_list][0],
                axis=0,
            )
            # print(self.state)
            # calcualte portfolio return
            # individual stocks' return * weight
            portfolio_return = sum(
                ((self.data.close.values / last_day_memory.close.values) - 1) * weights
            )
            # update portfolio value
            new_portfolio_value = self.portfolio_value * (1 + portfolio_return)
            self.portfolio_value = new_portfolio_value

            # save into memory
            self.portfolio_return_memory.append(portfolio_return)
            self.date_memory.append(self.data.date.unique()[0])
            self.asset_memory.append(new_portfolio_value)

            # the reward is the new portfolio value or end portfolo value
            self.reward = new_portfolio_value
            # print("Step reward: ", self.reward)
            # self.reward = self.reward*self.reward_scaling

        return self.state, self.reward, self.terminal, False, {}

    def reset(
        self,
        *,
        seed=None,
        options=None,
    ):
        self.asset_memory = [self.initial_amount]
        self.day = 0
        self.data = self.df.loc[self.day, :]
        # load states
        self.covs = self.data["cov_list"].values[0]
        self.state = np.append(
            [self.covs],
            [self.data[tech].values.tolist() for tech in self.tech_indicator_list][0],
            axis=0,
        )
        self.portfolio_value = self.initial_amount
        # self.cost = 0
        # self.trades = 0
        self.terminal = False
        self.portfolio_return_memory = [0]
        self.actions_memory = [[1 / self.stock_dim] * self.stock_dim]
        self.date_memory = [self.data.date.unique()[0]]
        return self.state, {}

    def render(self, mode="human"):
        return self.state

    def softmax_normalization(self, actions):
        numerator = np.exp(actions)
        denominator = np.sum(np.exp(actions))
        softmax_output = numerator / denominator
        return softmax_output

    def save_asset_memory(self):
        date_list = self.date_memory
        portfolio_return = self.portfolio_return_memory
        # print(len(date_list))
        # print(len(asset_list))
        df_account_value = pd.DataFrame(
            {"date": date_list, "daily_return": portfolio_return}
        )
        return df_account_value

    def save_action_memory(self):
        # date and close price length must match actions length
        date_list = self.date_memory
        df_date = pd.DataFrame(date_list)
        df_date.columns = ["date"]

        action_list = self.actions_memory
        df_actions = pd.DataFrame(action_list)
        df_actions.columns = self.data.tic.values
        df_actions.index = df_date.date
        # df_actions = pd.DataFrame({'date':date_list,'actions':action_list})
        return df_actions

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_sb_env(self):
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs
#%%
# ============================================================

import os

DATA_ROOT = os.environ.get(
    'DATA_ROOT',
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'data'),
)
OPTION_DAILY_DIR = os.path.join(DATA_ROOT, 'options', 'daily')
CONTRACT_FILE = os.path.join(DATA_ROOT, 'options', 'contracts', 'all_option_contracts.csv')
ETF_FILE = os.path.join(DATA_ROOT, 'underlyings', 'etf', '510300.SH.csv')

start_date = '2022-07-01'
end_date = '2023-05-31'


def load_contracts():
    """Read option contract info and keep CSI 300 ETF options (underlying OP510300.SH)"""
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
    return df[['code', 'name', 'contract_type', 'underlying_symbol', 'exercise_price', 'expire_date']]


def load_options_bar(date1, date2):
    """Read daily option quotes within the given date range (CSI 300 ETF options)"""
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
    """Read the CSI 300 ETF benchmark price"""
    df = pd.read_csv(ETF_FILE, encoding='utf-8-sig')
    df = df.rename(columns={'trade_date': 'date'})
    df['date'] = df['date'].apply(
        lambda x: datetime.strftime(pd.Timestamp(str(int(x))), '%Y-%m-%d')
    )
    return df[['date', 'close']].sort_values('date').reset_index(drop=True)


def calculate_hv(base_df, window=20):
    """Compute historical volatility (20-day annualised)
    std of daily close-to-close returns * sqrt(252)
    """
    df = base_df.copy()
    df['ret'] = df['close'].pct_change()
    df['hv'] = df['ret'].rolling(window=window).std() * np.sqrt(252)
    return df[['date', 'hv']].dropna()


def select_contracts(options_bar, options_info, base_df, start_date, end_date):
    """
    Option contract selection (identical to the original code logic):
    1. Sort by expiry date to determine roll dates
    2. For each roll cycle pick 3 calls (CO) and 3 puts (PO) with the largest turnover, sorted by strike
    3. Assign tic labels: b1,b2,b3 (calls) + s1,s2,s3 (puts)
    4. On non-roll days keep the contracts of the previous cycle (forward fill)
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
    hs300_df['base_p'] = 0

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
            tdf_co = tdf[tdf['contract_type'] == 'CO'].copy()
            tdf_co = tdf_co.loc[tdf_co['money'].nlargest(3).index].sort_values(by='exercise_price')
            tdf_co['tic'] = ['b1', 'b2', 'b3']
            tdf_po = tdf[tdf['contract_type'] == 'PO'].copy()
            tdf_po = tdf_po.loc[tdf_po['money'].nlargest(3).index].sort_values(by='exercise_price')
            tdf_po['tic'] = ['s1', 's2', 's3']
            tdf_sel = pd.concat([tdf_co, tdf_po])
            tdf_sel['base_p'] = base_df.loc[base_df['date'] == first_date, 'close'].values[0]
            tdate_val = tdf_sel['date'].iloc[0]
            mask = hs300_df['date'] == tdate_val
            hs300_df.loc[mask] = tdf_sel.values

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

        tdf_co = tdf[tdf['contract_type'] == 'CO'].copy()
        if len(tdf_co) < 3:
            continue
        tdf_co = tdf_co.loc[tdf_co['money'].nlargest(3).index].sort_values(by='exercise_price')
        tdf_co['tic'] = ['b1', 'b2', 'b3']

        tdf_po = tdf[tdf['contract_type'] == 'PO'].copy()
        if len(tdf_po) < 3:
            continue
        tdf_po = tdf_po.loc[tdf_po['money'].nlargest(3).index].sort_values(by='exercise_price')
        tdf_po['tic'] = ['s1', 's2', 's3']

        tdf_sel = pd.concat([tdf_co, tdf_po])
        tdf_sel['base_p'] = base_df.loc[base_df['date'] == tdate, 'close'].values[0]
        tdate_val = tdf_sel['date'].iloc[0]

        mask = hs300_df['date'] == tdate_val
        hs300_df.loc[mask] = tdf_sel.values

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
                tdf_co = tdf[tdf['contract_type'] == 'CO'].copy()
                tdf_co = tdf_co.loc[tdf_co['money'].nlargest(3).index].sort_values(by='exercise_price')
                tdf_co['tic'] = ['b1', 'b2', 'b3']
                tdf_po = tdf[tdf['contract_type'] == 'PO'].copy()
                tdf_po = tdf_po.loc[tdf_po['money'].nlargest(3).index].sort_values(by='exercise_price')
                tdf_po['tic'] = ['s1', 's2', 's3']
                tdf_sel = pd.concat([tdf_co, tdf_po])
                base_val = base_df.loc[base_df['date'] == last_turnover, 'close'].values
                tdf_sel['base_p'] = base_val[0] if len(base_val) > 0 else 0
                tdate_val = tdf_sel['date'].iloc[0]
                mask = hs300_df['date'] == tdate_val
                hs300_df.loc[mask] = tdf_sel.values

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
                base_val = base_df.loc[base_df['date'] == current_date, 'close'].values
                hs300_df.loc[i, 'base_p'] = base_val[0] if len(base_val) > 0 else 0

    hs300_df = hs300_df.dropna(subset=['code']).reset_index(drop=True)

    return hs300_df, expire_dates, turnover_dates


print("[Data loading] Loading Tushare option data...")
base_df = load_base_etf()
options_bar = load_options_bar(start_date, end_date)
options_info = load_contracts()

print(f"  Option daily bars: {len(options_bar)} records, {options_bar['code'].nunique()} contracts")
print(f"  Contract info: {len(options_info)} records")
print(f"  Benchmark ETF: {len(base_df)} trading days")

hs300_df, expire_dates, turnover_dates = select_contracts(
    options_bar, options_info, base_df, start_date, end_date
)

print(f"  After selection: {len(hs300_df)} rows, {hs300_df['date'].nunique()} trading days, 6 contracts/day")
print(f"  Expiry dates: {len(expire_dates)}, roll dates: {len(turnover_dates)}")

hs300_df['diff_days'] = hs300_df.apply(
    lambda x: (datetime.strptime(str(x.expire_date), '%Y-%m-%d') - datetime.strptime(str(x.date), '%Y-%m-%d')).days + 1, axis=1
)

hv_df = calculate_hv(base_df)
print(f"  HV data: {len(hv_df)} rows")
print("[Data loading] Done!")
exit()

#%%
expire_dates
#%%
hs300_df
#%%
import os

def check_and_make_directories(directories: list[str]):
    for directory in directories:
        if not os.path.exists("./" + directory):
            os.makedirs("./" + directory)

DATA_SAVE_DIR, TRAINED_MODEL_DIR,TENSORBOARD_LOG_DIR,RESULTS_DIR = 'datasets', 'trained_models', 'tensorboard_log', 'results'

check_and_make_directories([DATA_SAVE_DIR, TRAINED_MODEL_DIR, TENSORBOARD_LOG_DIR, RESULTS_DIR])
#%%
# hs300_df = get_data_options_data(start_date, end_date, '1000%%')
# hs300_df['date'] = hs300_df.date.apply(lambda x : str(x))
# hs300_df = hs300_df.rename(columns={'code':'tic'})

feature_engineer = FeatureEngineer(use_technical_indicator=False,use_turbulence=False, user_defined_feature=False)
processed_df = feature_engineer.preprocess_data(hs300_df)
turbulence_index = calculate_turbulence(hs300_df)
# add expire_dates
turbulence_index.turbulence = 0
turbulence_index.turbulence = turbulence_index.apply(lambda x : 100 if x.date in expire_dates else 0, axis=1)

processed_df = processed_df.merge(turbulence_index, on='date')
processed_df = processed_df.merge(hv_df, on='date')
# processed_df['cov_list'] = processed_df['turbulence']
processed_df.index = [int(i/6) for i in list(processed_df.index)]

train_data = processed_df.loc[processed_df.date<'2023-04-01', : ]
test_data = processed_df.loc[processed_df.date>='2023-04-01', : ]

test_data.index = test_data['date'].factorize()[0]
#%%
train_data.index.unique()
#%%
len(train_data),  len(test_data)

#%%
train_data
#%%
test_data
#%%
INDICATORS = ['exercise_price', 'base_p', 'diff_days', 'hv']
stock_dimension = len(train_data.tic.unique())
state_space = 1+ 2*stock_dimension + len(INDICATORS)*stock_dimension

buy_cost_list = sell_cost_list = [0.0001] * stock_dimension

num_stock_shares = [0] * stock_dimension

env_kwargs = {
    'hmax': 30000,
    'initial_amount': 60000,
    'num_stock_shares':num_stock_shares,
    'buy_cost_pct':buy_cost_list,
    'sell_cost_pct':sell_cost_list,
    'state_space': state_space,
    'stock_dim': stock_dimension,
    'tech_indicator_list':INDICATORS ,
    'action_space': stock_dimension,
    'reward_scaling': 1e-1,
    'turbulence_threshold' : 90,
}

e_train_gym = StockTradingEnv(df = train_data, **env_kwargs)
e_test_gym = StockTradingEnv(df = test_data, **env_kwargs)

#%%
env_train, _ = e_train_gym.get_sb_env()
print(env_train)
#%%
agent = DRLAgent(env=env_train)

if_using_a2c = True
if_using_ddpg = True
if_using_ppo = True
if_using_td3 = True
if_using_sac = True
#%%
# a2c
agent = DRLAgent(env=env_train)
model_a2c = agent.get_model('a2c')

trained_a2c = agent.train_model(model=model_a2c, tb_log_name='a2c', total_timesteps=50000)
trained_a2c.save('./results/a2c/agent_a2c')
#%%
# ddpg 
agent = DRLAgent(env=env_train)
model_ddpg = agent.get_model('ddpg')

trained_ddpg = agent.train_model(model=model_ddpg, tb_log_name='ddpg', total_timesteps=50000)
trained_ddpg.save('./results/ddpg/agent_ddpg')
#%%
# ppo
agent = DRLAgent(env=env_train)
ppo_params = {
    'n_steps':2048,
    'ent_coef':0.01,
    'learning_rate':0.00025,
    'batch_size':128,
}
model_ppo = agent.get_model('ppo', model_kwargs=ppo_params)

trained_ppo = agent.train_model(model=model_ppo, tb_log_name='ppo', total_timesteps=200000)
trained_ppo.save('./results/ppo/agent_ppo')

#%%
# td3
agent = DRLAgent(env=env_train)
td3_params = {
    'batch_size':50,
    'buffer_size':100000,
    'learning_rate':0.001
}
model_td3 = agent.get_model('td3', model_kwargs=td3_params)

trained_td3 = agent.train_model(model=model_td3, tb_log_name='td3', total_timesteps=50000)
trained_td3.save('./results/td3/agent_td3')
#%%
# sac
agent = DRLAgent(env=env_train)
sac_params = {
    'batch_size':50,
    'buffer_size':100000,
    'learning_rate':0.0001,
    'learning_starts':100,
    'ent_coef':'auto_0.1',
}
model_sac = agent.get_model('sac', model_kwargs=sac_params)

trained_sac = agent.train_model(model=model_sac, tb_log_name='sac', total_timesteps=70000)
trained_sac.save('./results/sac/agent_sac')
#%%
# prediction
# e_test_gym = StockTradingEnv(df=test_data,  **env_kwargs)

df_account_value_a2c , df_actions_a2c =  DRLAgent.DRL_prediction(model=model_a2c, environment=e_test_gym)
df_account_value_ddpg,df_actions_ddpg = DRLAgent.DRL_prediction(model=model_ddpg, environment=e_test_gym)
df_account_value_ppo,df_actions_ppo = DRLAgent.DRL_prediction(model=model_ppo, environment=e_test_gym)
df_account_value_td3,df_actions_td3 = DRLAgent.DRL_prediction(model=model_td3, environment=e_test_gym)
df_account_value_sac,df_actions_sac = DRLAgent.DRL_prediction(model=model_sac, environment=e_test_gym)


#%%
result = pd.DataFrame(
    {
        "a2c": df_account_value_a2c["account_value"],
        "ddpg": df_account_value_ddpg["account_value"],
        "ppo": df_account_value_ppo["account_value"],
        "td3": df_account_value_td3["account_value"],
        "sac": df_account_value_sac["account_value"],
    }
)
result.index = df_account_value_a2c['date'].to_list()
result
#%%
# df_account_value_td3,df_actions_td3 = DRLAgent.DRL_prediction(model=model_td3, environment=e_train_gym)
# df_account_value_td3,df_actions_td3 
#%%
def processed_df_for_mvo(df):
    return df.pivot(index='date', columns='tic', values='close')

def StockReturnsComputing(StockPrice, Rows, Columns):
    StockReturn = np.zeros([Rows-1, Columns])
    for j in range(Columns):
        for i in range(Rows-1):
            StockReturn[i,j] = ((StockPrice[i+1, j] - StockPrice[i, j]) / StockPrice[i, j]) * 100
    return StockReturn

StockData = processed_df_for_mvo(train_data)
TestData = processed_df_for_mvo(test_data)

TestData.to_numpy()

#%%
#  compute asset returns 
arStockPrices = np.asarray(StockData)
[Rows, Cols] = arStockPrices.shape
arReturns = StockReturnsComputing(arStockPrices, Rows, Cols)

meanReturns = np.mean(arReturns, axis=0)
covReturns = np.cov(arReturns, rowvar=False)

np.set_printoptions(precision=3, suppress=True)

print('mean returns of assets in k-portfolio 1\n', meanReturns)
print('variance-covariance matrix of returns\n', covReturns)

#%%
# add base index 
train_start_date = '2022-07-01'
train_end_date = '2023-03-31'
test_start_date = '2023-04-03'
test_end_date = '2023-05-31'

#%%
import copy 
df_base = copy.deepcopy(base_df[['date', 'close']])
df_base = df_base.loc[(df_base['date']>=test_start_date) &(df_base['date']<=test_end_date),: ]
df_base.reset_index(level = 0, drop=True, inplace=True)
fst_day = df_base['close'][0]
base = pd.merge(df_base['date'], df_base['close'].div(fst_day).mul(60000),
                how='outer', left_index=True, right_index=True).set_index('date')

#%%
result_c = pd.merge(result, base,
                how='outer', left_index=True, right_index=True)
#%%
plt.rcParams['figure.figsize'] = (15,5)
plt.subplot(1,1,1)

# result.plot()
p_df = result_c
plt.plot(p_df.close.values, label='base')
plt.plot(p_df.a2c.values, '--', label='a2c',linewidth=0.8, marker='1')
plt.plot(p_df.ddpg.values,'--', label='ddpg',linewidth=0.8, marker='2')
plt.plot(p_df.ppo .values,'--', label='ppo',linewidth=0.8, marker='+')
plt.plot(p_df.td3.values,'--', label='td3',linewidth=0.8, marker='*')
plt.plot(p_df.sac.values,'--', label='sac',linewidth=0.8, marker='o')

# x_ticks  = p_df.index.to_list()
# plt.xticks(range(0, len(x_ticks), 7), x_ticks.index[::7], rotation=70);
plt.legend()
plt.ylabel('price')

#%%
result_c.columns
#%%
result_c['a2c'][0]
#%%
# 
from empyrical import max_drawdown,sharpe_ratio

result_c['a2c_r'] = result_c['a2c'].pct_change()
result_c['ddpg_r'] =result_c['ddpg'].pct_change()
result_c['ppo_r'] =result_c['ppo'].pct_change()
result_c['td3_r'] = result_c['td3'].pct_change()
result_c['sac_r'] = result_c['sac'].pct_change()
result_c['close_r'] = result_c['close'].pct_change()

result_c['a2c_c'] = result_c['a2c']/result_c['a2c'][0]
result_c['ddpg_c'] = result_c['ddpg']/result_c['ddpg'][0]
result_c['ppo_c'] = result_c['ppo']/result_c['ppo'][0]
result_c['td3_c'] = result_c['td3']/result_c['td3'][0]
result_c['sac_c'] = result_c['sac']/result_c['sac'][0]
result_c['close_c'] = result_c['close']/result_c['close'][0]
#%%
# a2c ddpg ppo td3 sac
a2c_pf = result_c['a2c_c'][-1:].values[0]
a2c_md = max_drawdown(result_c['a2c_r'])
a2c_sr = sharpe_ratio(result_c['a2c_r'])

ddpg_pf = result_c['ddpg_c'][-1:].values[0]
ddpg_md = max_drawdown(result_c['ddpg_r'])
ddpg_sr = sharpe_ratio(result_c['ddpg_r'])

ppo_pf = result_c['ppo_c'][-1:].values[0]
ppo_md = max_drawdown(result_c['ppo_r'])
ppo_sr = sharpe_ratio(result_c['ppo_r'])

td3_pf = result_c['td3_c'][-1:].values[0]
td3_md = max_drawdown(result_c['td3_r'])
td3_sr = sharpe_ratio(result_c['td3_r'])

sac_pf = result_c['sac_c'][-1:].values[0]
sac_md = max_drawdown(result_c['sac_r'])
sac_sr = sharpe_ratio(result_c['sac_r'])

close_pf = result_c['close_c'][-1:].values[0]
close_md = max_drawdown(result_c['close_r'])
close_sr = sharpe_ratio(result_c['close_r'])
#%%
res_df = pd.DataFrame({'pf':[a2c_pf,ddpg_pf,ppo_pf,td3_pf,sac_pf,close_pf],
'md':[a2c_md,ddpg_md,ppo_md,td3_md,sac_md,close_md],
'sr':[a2c_sr,ddpg_sr,ppo_sr,td3_sr,sac_sr,close_sr]})
#%%
res_df
#%%
df_actions_a2c.to_csv('./results/weights_1/a2c.csv',encoding='gbk')
df_actions_ddpg.to_csv('./results/weights_1/ddpg.csv',encoding='gbk')
df_actions_ppo.to_csv('./results/weights_1/ppo.csv',encoding='gbk')
df_actions_td3.to_csv('./results/weights_1/td3.csv',encoding='gbk')
df_actions_sac.to_csv('./results/weights_1/sac.csv',encoding='gbk')
#%%
df_account_value_a2c , df_actions_a2c =  DRLAgent.DRL_prediction(model=model_a2c, environment=e_train_gym)
df_account_value_ddpg,df_actions_ddpg = DRLAgent.DRL_prediction(model=model_ddpg, environment=e_train_gym)
df_account_value_ppo,df_actions_ppo = DRLAgent.DRL_prediction(model=model_ppo, environment=e_train_gym)
df_account_value_td3,df_actions_td3 = DRLAgent.DRL_prediction(model=model_td3, environment=e_train_gym)
df_account_value_sac,df_actions_sac = DRLAgent.DRL_prediction(model=model_sac, environment=e_train_gym)

#%%
df_base = copy.deepcopy(base_df[['date', 'close']])
df_base = df_base.loc[(df_base['date']>=train_start_date) &(df_base['date']<=train_end_date),: ]
df_base.reset_index(level = 0, drop=True, inplace=True)
fst_day = df_base['close'][0]
base = pd.merge(df_base['date'], df_base['close'].div(fst_day).mul(60000),
                how='outer', left_index=True, right_index=True).set_index('date')

#%%
df_account_value_a2c
#%%
result = pd.DataFrame(
    {
        "a2c": df_account_value_a2c["account_value"],
        "ddpg": df_account_value_ddpg["account_value"],
        "ppo": df_account_value_ppo["account_value"],
        "td3": df_account_value_td3["account_value"],
        "sac": df_account_value_sac["account_value"],
    }
)
result.index = df_account_value_a2c['date'].to_list()

result_c = pd.merge(result, base,
                how='outer', left_index=True, right_index=True)
#%%
plt.rcParams['figure.figsize'] = (15,5)
plt.subplot(1,1,1)

# result.plot()
p_df = result_c
plt.plot(p_df.close.values, label='base')
plt.plot(p_df.a2c.values, '--', label='a2c',linewidth=0.8, marker='1')
plt.plot(p_df.ddpg.values,'--', label='ddpg',linewidth=0.8, marker='2')
plt.plot(p_df.ppo .values,'--', label='ppo',linewidth=0.8, marker='+')
plt.plot(p_df.td3.values,'--', label='td3',linewidth=0.8, marker='*')
plt.plot(p_df.sac.values,'--', label='sac',linewidth=0.8, marker='o')

# x_ticks  = p_df.index.to_list()
# plt.xticks(range(0, len(x_ticks), 7), x_ticks.index[::7], rotation=70);
plt.legend()
plt.ylabel('price')
#%%
result_c['a2c_r'] = result_c['a2c'].pct_change()
result_c['ddpg_r'] =result_c['ddpg'].pct_change()
result_c['ppo_r'] =result_c['ppo'].pct_change()
result_c['td3_r'] = result_c['td3'].pct_change()
result_c['sac_r'] = result_c['sac'].pct_change()
result_c['close_r'] = result_c['close'].pct_change()

result_c['a2c_c'] = result_c['a2c']/result_c['a2c'][0]
result_c['ddpg_c'] = result_c['ddpg']/result_c['ddpg'][0]
result_c['ppo_c'] = result_c['ppo']/result_c['ppo'][0]
result_c['td3_c'] = result_c['td3']/result_c['td3'][0]
result_c['sac_c'] = result_c['sac']/result_c['sac'][0]
result_c['close_c'] = result_c['close']/result_c['close'][0]
#%%
a2c_pf = result_c['a2c_c'][-1:].values[0]
a2c_md = max_drawdown(result_c['a2c_r'])
a2c_sr = sharpe_ratio(result_c['a2c_r'])

ddpg_pf = result_c['ddpg_c'][-1:].values[0]
ddpg_md = max_drawdown(result_c['ddpg_r'])
ddpg_sr = sharpe_ratio(result_c['ddpg_r'])

ppo_pf = result_c['ppo_c'][-1:].values[0]
ppo_md = max_drawdown(result_c['ppo_r'])
ppo_sr = sharpe_ratio(result_c['ppo_r'])

td3_pf = result_c['td3_c'][-1:].values[0]
td3_md = max_drawdown(result_c['td3_r'])
td3_sr = sharpe_ratio(result_c['td3_r'])

sac_pf = result_c['sac_c'][-1:].values[0]
sac_md = max_drawdown(result_c['sac_r'])
sac_sr = sharpe_ratio(result_c['sac_r'])

close_pf = result_c['close_c'][-1:].values[0]
close_md = max_drawdown(result_c['close_r'])
close_sr = sharpe_ratio(result_c['close_r'])

res_df = pd.DataFrame({'pf':[a2c_pf,ddpg_pf,ppo_pf,td3_pf,sac_pf,close_pf],
'md':[a2c_md,ddpg_md,ppo_md,td3_md,sac_md,close_md],
'sr':[a2c_sr,ddpg_sr,ppo_sr,td3_sr,sac_sr,close_sr]})
#%%
res_df
#%%
