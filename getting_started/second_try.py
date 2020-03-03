import numpy as np
import torch
import ray.rllib
import pybullet
import yaml
import ray
from ray import tune
import gym

ray.init()
tune.run(
    "DQN",
    stop = {"episode_reward_mean": 200},
    config = {
        "env": "CartPole-v0",
        "num_gpus": 0,
        "num_workers": 1,
        "lr": tune.grid_search([0.01, 0.001, 0.0001]),
        "eager": False,
    },
)

env = gym.make("CartPole-v0")
observation = env.reset()
for i in range(1000):
    env.render()
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    
    if done:
        observation = env.reset()
env.close()