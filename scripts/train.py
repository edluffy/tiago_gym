#! /usr/bin/env python
import rospy
import gym
import tiago_reach_env
from agents import dqn

import tensorflow as tf
tf.compat.v1.enable_eager_execution()



if __name__ == '__main__':
    rospy.init_node('tiago_gym')
    env = gym.make('TiagoReachEnv-v0')
    obs = env.reset()

    o_space = len(env.observation_space.sample())
    a_space = env.action_space.n
    agent = dqn.DQN(env, input_size=o_space, output_size=a_space)
    agent.run(episodes=1000)

    #for _ in range(10):
    #    action = env.action_space.sample()
    #    obs, reward, done, _ = env.step(action)
    #    print('action:', action)
    #    print('obs:', obs)
    #    print('reward:', reward)
    #    print('done:', done)

    #    if done:
    #        break

