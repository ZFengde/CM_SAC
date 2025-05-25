import gymnasium as gym
import numpy as np
import torch as th
import pandas as pd

from stable_baselines3 import PPO, TD3, DDPG

env = gym.make("Ant-v4")
env.reset()

for i in range(3000):
    action = env.action_space.sample()
    new_obs, reward, done, info, _ = env.step(action)
    print(reward)
    if done:
        obs = env.reset()
        t = 0