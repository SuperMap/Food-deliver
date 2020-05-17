# -*- coding: utf-8 -*-
# @Author  : woleto
# @Time    : 2020/5/11 10:41
import torch

from DeliverEnv import DeliverEnv
from RLBrain import DeepQNetwork

MODELSAVEDIR = 'model'

def runDeliver():
    step = 0
    for episode in range(300):
        # initial observation
        observation = env.reset()

        while True:
            # fresh env
            env.render(episode, step)

            # RL choose action based on observation
            action = RL.choose_action(observation)

            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)

            RL.store_transition(observation, action, reward, observation_)

            model = None
            if (step > 200) and (step % 5 == 0):
                model = RL.learn()

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                torch.save(model, MODELSAVEDIR + '/' + env.areaId + '.pkl')
                break
            step += 1

    # end of game
    print('deliver over')


if __name__ == "__main__":
    # maze game
    env = DeliverEnv('680507')
    RL = DeepQNetwork()
    runDeliver()