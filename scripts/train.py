#! /usr/bin/env python
import rospy
import gym
#from openai_ros.openai_ros_common import StartOpenAI_ROS_Environment
#from agents import dqn

#o_space = len(env.observation_space.sample())
#a_space = env.action_space.n
#agent = DQN(input_size=o_space, output_size=a_space)

import tiago_move_task

if __name__ == '__main__':
    rospy.init_node('tiago_gym')
    env = gym.make('TiagoMoveEnv-v0')
    obs = env.reset()

    for _ in range(5):
        action = env.action_space.sample()
        obs, reward, done, _ = env.step(action)

