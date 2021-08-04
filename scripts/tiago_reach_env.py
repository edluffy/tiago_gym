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
        self.obs_low = np.array([0.4, -0.5, 0.5])
        self.obs_high = np.array([0.6, 0.5, 0.7])

        self.observation_space = spaces.Box(self.obs_low, self.obs_high)

        # Action space
        #high = np.array([0.1, 0.1, 0.1])
        #self.action_space = spaces.Box(-high, high)
        self.action_space = spaces.Discrete(6)

        # randomize goal
        x = np.random.uniform(self.obs_low[0], self.obs_high[0])
        y = np.random.uniform(self.obs_low[1], self.obs_high[1])
        z = np.random.uniform(self.obs_low[2], self.obs_high[2])

        self.goal = np.round([x, y, z], 1)
        self.set_marker_points([self.goal], ns='goal')


    def _set_init_pose(self):
        """Sets the Robot in its init pose
        """
        x, y, z = 0.5, 0.0, 0.6
        roll, pitch, yaw = 0, np.radians(90), np.radians(90)
        self.send_arm_pose(x, y, z, roll, pitch, yaw)

    def _init_env_variables(self):
        """
        Inits variables needed to be initialised each time we reset at the start
        of an episode.
        :return:
        """
        self.action_count = 0

    def _set_action(self, action):
        """
        Move the robot based on the action variable given
        """
        self.action_count += 1
        #if type(action) == np.ndarray:
        #    action = action.tolist()

        if self.action_count >= 20:
            self.action_failed = True
            return

        plan = True
        if action == 1:
            plan = self.shift_arm_pose(0.1, 'x')
        elif action == 2:
            plan = self.shift_arm_pose(-0.1, 'x')
        elif action == 3:
            plan = self.shift_arm_pose(0.1, 'y')
        elif action == 4:
            plan = self.shift_arm_pose(-0.1, 'y')
        elif action == 5:
            plan = self.shift_arm_pose(0.1, 'z')
        elif action == 6:
            plan = self.shift_arm_pose(-0.1, 'z')

        #self.set_marker_points([[x, y, z]], ns='action')
        #self.send_arm_pose(x, y, z, roll, pitch, yaw)
        self.action_failed = not plan

    def _get_obs(self):
        """
        Here we define what sensor data of our robots observations
        To know which Variables we have acces to, we need to read the
        MyRobotEnv API DOCS
        :return: observations
        """
        x, y, z, _, _, _ = self.stored_arm_pose
        observations = [x, y, z]
        return observations

    def _is_done(self, observations):
        """
        Decide if episode is done based on the observations
        """
        delta = self.get_goal_delta(observations)

        if delta <= 0.05 or self.action_failed or rospy.is_shutdown():
            done = True
        else:
            done = False
        return done

    def _compute_reward(self, observations, done):
        """
        Return the reward based on the observations given
        """
        delta = self.get_goal_delta(observations)
        
        if self.action_failed:
            reward = -10
        elif done:
            reward = 20
        else:
            reward = -delta
        return reward
        
    # Internal TaskEnv Methods

    def get_goal_delta(self, obs):
        ee_xyz = np.array(obs)
        goal_xyz = np.array(self.goal)

        delta = np.linalg.norm(ee_xyz-goal_xyz)
        return delta

