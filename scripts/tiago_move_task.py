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
    """
    Observation:
        Type: Box(3)
        Num     Observation
        0       absolute pos of end effector
        1       relative pos of end effector to goal
        2       distance between gripper fingers

    Actions:
        Type: Box(4)
        Num     Action
        0       x-pos of end effector
        1       y-pos of end effector
        2       z-pos of end effector
        3       distance between gripper fingers

    Reward:

    """
    def __init__(self):
        super(TiagoMoveEnv, self).__init__()

        # Observation space
        high = np.array([0, 0])
        self.observation_space = spaces.Box(-high, high)

        # Action space
        low = np.array([0.0, -0.5, 0.7])
        high = np.array([0.8, 0.5, 0.7])
        self.action_space = spaces.Box(low, high)

        # Reward
        self.goal_pos = [0.2, 0.0, -1.5, 1.94, -1.57, -0.5, 0.0]


    def _set_init_pose(self):
        """Sets the Robot in its init pose
        """
        x, y, z = 0.8, 0.0, 0.7
        roll, pitch, yaw = 0, np.radians(90), np.radians(90)
        self.send_arm_pose(x, y, z, roll, pitch, yaw)


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
        if type(action) == np.ndarray:
            action = action.tolist()

        x, y, z = action
        roll, pitch, yaw = 0, np.radians(90), np.radians(90)
        self.send_arm_pose(x, y, z, roll, pitch, yaw)

    def _get_obs(self):
        """
        Here we define what sensor data of our robots observations
        To know which Variables we have acces to, we need to read the
        MyRobotEnv API DOCS
        :return: observations
        """

        observations = None
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
