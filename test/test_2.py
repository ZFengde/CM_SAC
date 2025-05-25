import gymnasium as gym
import numpy as np
import torch as th
import pandas as pd

from stable_baselines3 import PPO, TD3, DDPG

env = gym.make("Hopper-v4")

model = TD3("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=5000000)

vec_env = model.get_env()


obs = vec_env.reset()

for i in range(3000):
    action, _obs = model.predict(obs, deterministic=True)
    new_obs, reward, done, info = vec_env.step(action)
    if done:
        obs = vec_env.reset()
        t = 0