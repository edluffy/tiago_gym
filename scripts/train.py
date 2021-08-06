#! /usr/bin/env python
import rospy
import gym
import tiago_reach_env
from agents import dqn, ddpg
import tensorflow as tf

tf.compat.v1.enable_eager_execution()


# To run as fast as possible:
# roslaunch tiago_gym start_training.launch gazebo:=false
# gz physics -u 0 -s 0.0025; gz stats

if __name__ == '__main__':
    rospy.init_node('tiago_gym')
    env = gym.make('TiagoReachEnv-v0')

    #o_space = len(env.observation_space.sample())
    #a_space = env.action_space.n
    #agent = dqn.DQN(env, input_size=o_space, output_size=a_space, alpha=0.01, epsilon_decay=0.95)

    o_dims = len(env.observation_space.sample())
    a_dims = env.action_space.shape[0]
    a_high = env.action_space.high
    
    agent = ddpg.DDPG(env, o_dims, a_dims, a_high)
    agent.run(100000)
