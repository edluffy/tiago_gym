import rospy
import numpy as np
from gym import spaces
from gym.envs.registration import register
import tiago_env

max_episode_steps = 1000

register(
        id='TiagoReachEnv-v0',
        entry_point='tiago_reach_env:TiagoReachEnv',
        max_episode_steps=max_episode_steps,
    )
class TiagoReachEnv(tiago_env.TiagoEnv):
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
        super(TiagoReachEnv, self).__init__()

        # Observation space
        high = np.array([0, 0])
        self.observation_space = spaces.Box(-high, high)

        # Action space
        low = np.array([0.0, -0.5, 0.7])
        high = np.array([0.8, 0.5, 0.7])
        self.action_space = spaces.Box(low, high)

        # Reward

        self.goal = (1.0, 0.0, 0.7)
        self.update_marker(self.goal[0], self.goal[1], self.goal[2], ns='goal')


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

        self.update_marker(x, y, z, 'action')
        self.send_arm_pose(x, y, z, roll, pitch, yaw)

    def _get_obs(self):
        """
        Here we define what sensor data of our robots observations
        To know which Variables we have acces to, we need to read the
        MyRobotEnv API DOCS
        :return: observations
        """
        pos = self.get_arm_pose().position
        ee_xyz = np.array([pos.x, pos.y, pos.z])
        goal_xyz = np.array(self.goal)

        abs_pos = np.linalg.norm(ee_xyz)
        rel_pos = np.linalg.norm(ee_xyz-goal_xyz)

        observations = [abs_pos, rel_pos]

        return observations

    def _is_done(self, observations):
        """
        Decide if episode is done based on the observations
        """
        done = False
        if rospy.is_shutdown():
            done = True
        return done

    def _compute_reward(self, observations, done):
        """
        Return the reward based on the observations given
        """
        if self._is_done(observations):
            reward = 10
        else:
            reward = -1.0

        rospy.loginfo(">>>REWARD>>>"+str(reward))
        return reward
        
    # Internal TaskEnv Methods
