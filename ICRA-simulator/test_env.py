from spinup import ppo_tf1 as ppo
import tensorflow as tf
import gym
import ICRA_simulator
import random

env = gym.make('ICRA_simulator:icra-simulator-v0',car_num = 1, robot_id = 1, render = True, record = False)


def ppo_test():
    # ac_kwargs = dict(hidden_sizes=[64,64])

    # logger_kwargs = dict(output_dir='path/to/output_dir', exp_name='experiment_name')

    ppo(env_fn=env, steps_per_epoch=50, epochs=10)

    

def demo():
    total_reward = 0
    state = env.reset()
    steps = 0
    while True:
        if env.get_order():
            return
        action = env.orders
        state, reward, done, info = env.step(action)
        steps += 1
        total_reward += reward
        if steps % 20 == 0 or done:
           print("step {} total_reward {:+0.2f}".format(steps, total_reward))
        if done:
           return total_reward
if __name__ =="__main__":
    demo()