#! /usr/bin/env python
import rospy
import gym
from openai_ros.openai_ros_common import StartOpenAI_ROS_Environment
#from agents import dqn

#o_space = len(env.observation_space.sample())
#a_space = env.action_space.n
#agent = DQN(input_size=o_space, output_size=a_space)

from openai_ros.task_envs.cartpole_stay_up import stay_up


if __name__ == '__main__':
    rospy.init_node('tiago_robot_train')
    task_and_robot_environment_name = rospy.get_param('/fetch/task_and_robot_environment_name')
    env = StartOpenAI_ROS_Environment(task_and_robot_environment_name)
    obs = env.reset()

    rate = rospy.Rate(30)

    for _ in range(10):
        env.render()
        action = env.action_space.sample()
        obs, reward, done, _ = env.step(action)

