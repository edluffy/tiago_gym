import rospy
import numpy as np
from gym import spaces
from gym.envs.registration import register
import tiago_env

max_episode_steps = 1000

register(
        id='TiagoSimpleEnv-v0',
        entry_point='tiago_simple_env:TiagoSimpleEnv',
        max_episode_steps=max_episode_steps,
    )
class TiagoSimpleEnv(tiago_env.TiagoEnv):
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
        super(TiagoSimpleEnv, self).__init__()

        # Observation space
        self.obs_low = np.array([0.4, -0.25, 0.65])
        self.obs_high = np.array([0.7, 0.25, 0.75])

        self.observation_space = spaces.Box(self.obs_low, self.obs_high)

        # Action space
        #high = np.array([0.1, 0.1, 0.1])
        #self.action_space = spaces.Box(-high, high)
        self.action_space = spaces.Discrete(6)

        self.obs_points = {'x': [], 'y': [], 'z':[]}
        for x in np.arange(self.obs_low[0], self.obs_high[0]+0.1, 0.1):
            for y in np.arange(self.obs_low[1], self.obs_high[1]+0.1, 0.1):
                for z in np.arange(self.obs_low[2], self.obs_high[2]+0.1, 0.1):
                    self.obs_points['x'].append(x)
                    self.obs_points['y'].append(y)
                    self.obs_points['z'].append(z)
        self.visualize_points(self.obs_points['x'], self.obs_points['y'], self.obs_points['z'], ns='obs')

        # randomize goal
        x = np.random.uniform(self.obs_low[0], self.obs_high[0])
        y = np.random.uniform(self.obs_low[1], self.obs_high[1])
        z = np.random.uniform(self.obs_low[2], self.obs_high[2])

        x = self.obs_points['x'][np.abs(np.asarray(self.obs_points['x'])-x).argmin()]
        y = self.obs_points['y'][np.abs(np.asarray(self.obs_points['y'])-y).argmin()]
        z = self.obs_points['z'][np.abs(np.asarray(self.obs_points['z'])-z).argmin()]

        self.goal = [x, y, z]

    def _set_init_pose(self):
        """Sets the Robot in its init pose
        """
        self.visualize_points(*self.goal, ns='goal')
        x, y, z = 0.5, 0.15, 0.65
        roll, pitch, yaw = 0, np.radians(90), np.radians(90)
        self.set_gripper_joints(0.001, 0.001)
        self.set_arm_pose(x, y, z, roll, pitch, yaw)

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

        x, y, z, _, _, _ = self.stored_arm_state

        plan = True
        if action == 1:
            x += 0.1
        elif action == 2:
            x -= 0.1
        elif action == 3:
            y += 0.1
        elif action == 4:
            y -= 0.1
        elif action == 5:
            z += 0.1
        elif action == 6:
            z -= 0.1
        roll, pitch, yaw = 0, np.radians(90), np.radians(90)

        x = self.obs_points['x'][np.abs(np.asarray(self.obs_points['x'])-x).argmin()]
        y = self.obs_points['y'][np.abs(np.asarray(self.obs_points['y'])-y).argmin()]
        z = self.obs_points['z'][np.abs(np.asarray(self.obs_points['z'])-z).argmin()]

        self.visualize_points(x, y, z, ns='action')
        self.action_failed = not self.set_arm_pose(x, y, z, roll, pitch, yaw)


    def _get_obs(self):
        """
        Here we define what sensor data of our robots observations
        To know which Variables we have acces to, we need to read the
        MyRobotEnv API DOCS
        :return: observations
        """
        x, y, z, _, _, _ = self.stored_arm_state
        observations = [x, y, z]
        return observations

    def _is_done(self, observations):
        """
        Decide if episode is done based on the observations
        """
        if self.action_count >= 50 or rospy.is_shutdown():
            done = True
        else:
            done = False
        return done

    def _compute_reward(self, observations, done):
        """
        Return the reward based on the observations given
        """
        _, rel, = self.calculate_distances()
        
        if rel <= 0.05:
            reward = 10
        else:
            reward = -rel

        return reward
        
    # Internal TaskEnv Methods

    def calculate_distances(self):
        ee_xyz = np.array(self.stored_arm_state[:3])
        goal_xyz = np.array(self.goal)

        abs = np.linalg.norm(ee_xyz)
        rel = np.linalg.norm(ee_xyz-goal_xyz)
        return abs, rel


