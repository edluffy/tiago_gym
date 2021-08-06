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
        0       absolute pos of gripper
        1       relative pos of gripper to goal
        2       actual distance between gripper fingers

    Actions:
        Type: Box(4)
        Num     Action
        0       x-pos of gripper
        1       y-pos of gripper
        2       z-pos of gripper
        3       desired distance between gripper fingers

    Rewards (Sparse!):
        Goal not reached yet: -1
        Goal reached: 20

    Done:
        When episode length > 50

    """
    def __init__(self):
        super(TiagoReachEnv, self).__init__()

        # Observation space TODO:
        o_low =  np.array([0, 0, 0])
        o_high = np.array([0, 0, 0])

        self.observation_space = spaces.Box(o_low, o_high)

        # Action space TODO:
        a_high = np.array([1, 1, 1, 1])
        self.action_space = spaces.Box(-a_high, a_high)

        # randomize goal
        x, y, z = np.random.uniform(self.arm_workspace_low, self.arm_workspace_high)
        self.goal = np.round([x, y, z], 1)
        self.set_marker_points([self.goal], ns='goal')

        self.past_positions = []
        # 0.35, -0.25, 0.65
        # 0.75, -0.25, 0.65
        # 0.75, -0.25, 0.75

    def _set_init_pose(self):
        """Sets the Robot in its init pose
        """
        x, y, z = 0.5, 0.0, 0.7
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
        if type(action) == np.ndarray:
            action = action.tolist()

        dx, dy, dz, _ = action
        print(dx, dy, dz)
        x = self.stored_arm_pose[0] + dx*0.05
        y = self.stored_arm_pose[1] + dy*0.05
        z = self.stored_arm_pose[2] + dz*0.05
        roll, pitch, yaw = 0, np.radians(90), np.radians(90)

        if self.arm_pose_reachable(x, y, z):
            plan = self.send_arm_pose(x, y, z, roll, pitch, yaw)
        else:
            self.past_positions.append([x, y, z])
            self.set_marker_points(self.past_positions, ns='action')
            plan = False

        self.action_failed = not plan

    def _get_obs(self):
        """
        Here we define what sensor data of our robots observations
        To know which Variables we have acces to, we need to read the
        MyRobotEnv API DOCS
        :return: observations
        """
        abs, rel = self.calculate_distances()
        observations = [abs, rel, 0]
        return observations

    def _is_done(self, observations):
        """
        Decide if episode is done based on the observations
        """
        _, rel, _ = observations

        if rel <= 0.05 or self.action_count >= 50 or rospy.is_shutdown():
            done = True
        else:
            done = False
        return done

    # Internal TaskEnv Methods
    def _compute_reward(self, observations, done):
        """
        Return the reward based on the observations given
        """
        _, rel, _ = observations

        if done and rel <= 0.05:
            reward = 20
        elif self.action_failed:
            reward = -5
        else:
            reward = -rel

        return reward

    def calculate_distances(self):
        ee_xyz = np.array(self.stored_arm_pose[:3])
        goal_xyz = np.array(self.goal)

        abs = np.linalg.norm(ee_xyz)
        rel = np.linalg.norm(ee_xyz-goal_xyz)
        return abs, rel
