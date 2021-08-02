#! /usr/bin/env python
import rospy
import gym
import tiago_reach_env
from agents import dqn
import tensorflow as tf

tf.compat.v1.enable_eager_execution()


# To run as fast as possible:
# roslaunch tiago_gym start_training.launch gazebo:=false
# gz physics -u 0; gz stats

if __name__ == '__main__':
    rospy.init_node('tiago_gym')
    env = gym.make('TiagoReachEnv-v0')

    o_space = len(env.observation_space.sample())
    a_space = env.action_space.n
    agent = dqn.DQN(env, input_size=o_space, output_size=a_space, alpha=0.01, epsilon_decay=0.95)
    agent.run(episodes=50000, name='a=0.01,dec=0.95,fixed')
    #agent1 = dqn.DQN(env, input_size=o_space, output_size=a_space, alpha=0.01, epsilon_decay=0.98)
    #agent1.run(episodes=50000, name='a=0.01,dec=0.98,act=60')

    #agent2 = dqn.DQN(env, input_size=o_space, output_size=a_space, alpha=0.05, epsilon_decay=0.98)
    #agent2.run(episodes=500, name='a=0.05,dec=0.98')

    #agent3 = dqn.DQN(env, input_size=o_space, output_size=a_space, alpha=0.1, epsilon_decay=0.98)
    #agent3.run(episodes=500, name='a=0.1,dec=0.98')

    #agent4 = dqn.DQN(env, input_size=o_space, output_size=a_space, alpha=0.03, epsilon_decay=0.995)
    #agent4.run(episodes=5000000, name='a=0.03,dec=0.995')
