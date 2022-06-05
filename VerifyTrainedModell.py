import torch
import gym
import time
import random
from torch.nn.functional import normalize
import numpy as np

from AC import ActorCritic

if __name__ == '__main__':
    #Load trained modell from .pth File
    model = torch.load('trained_modell.pth')
    model.eval()
    env = gym.make("CartPole-v1")
    observation = env.reset()
    fails = 0
    for i in range(1000):
        env.render()
        state = torch.from_numpy(observation)
        policy, view = model(state)

        #In the trained modell we take the action with the highest value from the prohability distribution.
        if policy[0] > policy[1]:
            action = 0
        else:
            action = 1

        observation, reward, done, info = env.step(action) #Make predictions with the trained Model
        #observation, reward, done, info = env.step(random.randint(0,1)) #Compare Reward with random actions
        if done:
            fails += 1
            observation = env.reset()
    print("Fails: ", fails)
    time.sleep(3)
    env.close()
