# -*- coding: utf-8 -*-
# @Author  : woleto
# @Time    : 2020/5/11 10:41
from DeliverEnv import DeliverEnv
from RLBrain import DeepQNetwork


def runDeliver():
    step = 0
    for episode in range(300):
        # initial observation
        observation = env.reset()

        while True:
            # fresh env
            env.render()

            # RL choose action based on observation
            action = RL.choose_action(observation)

            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)

            RL.store_transition(observation, action, reward, observation_)

            if (step > 200) and (step % 5 == 0):
                RL.learn()

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                break
            step += 1

    # end of game
    print('deliver over')


if __name__ == "__main__":
    # maze game
    env = DeliverEnv('680507')
    RL = DeepQNetwork()
    runDeliver()