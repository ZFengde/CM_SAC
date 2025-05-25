import gymnasium as gym
import numpy as np
import bopu_env
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

env = gym.make('bopu-v1', csv_path = './output.csv') 

# The noise objects for TD3
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

model = TD3("MlpPolicy", env, action_noise=action_noise, verbose=1)
model.learn(total_timesteps=1000000, log_interval=1)
model.save("td3_bopu")
vec_env = model.get_env()

del model # remove to demonstrate saving and loading

model = TD3.load("td3_bopu")

obs = vec_env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
