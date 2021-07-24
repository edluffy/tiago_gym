#! /usr/bin/env python
import rospy
import actionlib
from openai_ros import robot_gazebo_env
from openai_ros.openai_ros_common import ROSLauncher
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from trajectory_msgs.msg import JointTrajectoryPoint

class TiagoEnv(robot_gazebo_env.RobotGazeboEnv):
    """Tiago Steel environment
    """
    def __init__(self):
        rospy.logdebug("========= In Tiago Env")

        #ROSLauncher(rospackage_name="tiago_gym",
        #            launch_file_name="tiago_steel_gazebo.launch",
        #            ros_ws_abspath=ros_ws_abspath)

        # Gotten via 'rosservice call /controller_manager/list_controllers | grep name'
        self.controllers_list = ['joint_state_controller',
                                 'imu_sensor_controller',
                                 'gripper_controller',
                                 'torso_controller',
                                 'head_controller',
                                 'arm_controller',
                                 'mobile_base_controller']
        
        # Does not use namespace, 'rostopic list | grep controller'
        self.robot_name_space = ""

        # Whether to reset controllers when a new episode starts
        reset_controls_bool = False
        
        # Parent class init
        super(TiagoEnv, self).__init__(controllers_list=self.controllers_list,
                                       robot_name_space=self.robot_name_space,
                                       reset_controls=reset_controls_bool)
        
        self.gazebo.unpauseSim()
        self.controllers_object.reset_controllers()

        # Set up action clients and wait for ready
        self.arm_client = actionlib.SimpleActionClient('arm_controller/follow_joint_trajectory', FollowJointTrajectoryAction)
        self.arm_goal = FollowJointTrajectoryGoal()
        self.arm_joint_names = ['arm_1_joint', 'arm_2_joint', 'arm_3_joint',
                                'arm_4_joint', 'arm_5_joint', 'arm_6_joint', 'arm_7_joint']
        # TODO: add other clients : '/head_controller', '/gripper_controller'

        self._check_all_systems_ready()
        self.move_to_init_pose()
        self.gazebo.pauseSim()

    # Methods needed by the RobotGazeboEnv
    # ----------------------------
    def _check_all_systems_ready(self):
        """
        Checks that all the sensors, publishers and other simulation systems are
        operational.
        """
        self.arm_client.wait_for_server()
        # TODO: head_client.wait_for_server()
        # TODO: gripper_client.wait_for_server()
        rospy.logdebug("ALL CLIENTS READY")
        return True

    # TiagoEnv virtual methods
    # ----------------------------

    def _arm_joints_callback(self, feedback):
        rospy.logdebug('arm joints got feedback')

    def get_arm_joints(self):
        return self.arm_joints

    def move_to_init_pose(self):
        self.set_arm_trajectory([[0.2, 0.0, -1.5, 1.94, -1.57, -0.5, 0.0]])
    
    def set_arm_trajectory(self, positions):
        count = len(positions)
        self.arm_goal.trajectory.joint_names = self.arm_joint_names
        self.arm_goal.trajectory.points = [JointTrajectoryPoint() for _ in range(count)]

        for n in range(count):
            self.arm_goal.trajectory.points[n].positions = positions[n]
            self.arm_goal.trajectory.points[n].time_from_start = rospy.Duration(2.0)

        self.arm_client.send_goal(self.arm_goal)
        self.arm_client.wait_for_result()

        #self.arm_joints = self.arm_client.get_result()
        print(self.arm_client.get_result().points[0])

    
    # Methods that the TrainingEnvironment will need to define here as virtual
    # because they will be used in RobotGazeboEnv GrandParentClass and defined in the
    # TrainingEnvironment.
    # ----------------------------
    def _set_init_pose(self):
        """Sets the Robot in its init pose
        """
        raise NotImplementedError()
    
    
    def _init_env_variables(self):
        """Inits variables needed to be initialised each time we reset at the start
        of an episode.
        """
        raise NotImplementedError()

    def _compute_reward(self, observations, done):
        """Calculates the reward to give based on the observations given.
        """
        raise NotImplementedError()

    def _set_action(self, action):
        """Applies the given action to the simulation.
        """
        raise NotImplementedError()

    def _get_obs(self):
        raise NotImplementedError()

    def _is_done(self, observations):
        """Checks if episode done based on observations given.
        """
        raise NotImplementedError()
        
    # Methods that the TrainingEnvironment will need.
    # ----------------------------
rospy.init_node('trajectory_controller')
test = TiagoEnv()
