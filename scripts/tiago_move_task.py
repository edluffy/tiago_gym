import rospy
import numpy as np
from gym import spaces
from gym.envs.registration import register
import tiago_env

max_episode_steps = 1000

register(
        id='TiagoMoveEnv-v0',
        entry_point='tiago_move_task:TiagoMoveEnv',
        max_episode_steps=max_episode_steps,
    )

class TiagoMoveEnv(tiago_env.TiagoEnv):
    def __init__(self):
        
        # Only variable needed to be set here
        #number_actions = rospy.get_param('/my_robot_namespace/n_actions')
        number_actions = 14
        self.action_space = spaces.Discrete(number_actions)

        super(TiagoMoveEnv, self).__init__()
        
        joint_limits = self.get_joint_limits()
        high = np.array([limit[0] for limit in joint_limits])
        low  = np.array([limit[1] for limit in joint_limits])

        self.observation_space = spaces.Box(low, high)
        
        # Variables that we retrieve through the param server, loded when launch training launch.
        self.position_delta = 1.0
        


    def _set_init_pose(self):
        """Sets the Robot in its init pose
        """
        self.set_joint_positions([0.2, 0.0, -1.5, 1.94, -1.57, -0.5, 0.0])
        return True


    def _init_env_variables(self):
        """
        Inits variables needed to be initialised each time we reset at the start
        of an episode.
        :return:
        """
        # TODO


    def _set_action(self, action):
        """
        Move the robot based on the action variable given
        """

        joint_positions = list(self.get_joint_positions())
        if action < 7:
            joint_positions[action] += self.position_delta
        else:
            action -= 7
            joint_positions[action] -= self.position_delta

        self.set_joint_positions(joint_positions)

    def _get_obs(self):
        """
        Here we define what sensor data of our robots observations
        To know which Variables we have acces to, we need to read the
        MyRobotEnv API DOCS
        :return: observations
        """
        joint_positions = self.get_joint_positions()
        observations = joint_positions

        rospy.logdebug("OBSERVATIONS====>>>>>>>"+str(observations))

        return observations

    def _is_done(self, observations):
        """
        Decide if episode is done based on the observations
        """
        done = False
        return done

    def _compute_reward(self, observations, done):
        """
        Return the reward based on the observations given
        """
        reward = -1.0
        if done:
            reward += self.reached_goal_reward
        rospy.loginfo(">>>REWARD>>>"+str(reward))
        return reward
        
    # Internal TaskEnv Methods
