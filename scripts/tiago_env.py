#! /usr/bin/env python
import sys
import rospy
import moveit_commander
import numpy as np
from openai_ros import robot_gazebo_env
from openai_ros.openai_ros_common import ROSLauncher
from tf.transformations import quaternion_from_euler
from geometry_msgs.msg import Pose
from visualization_msgs.msg import Marker

class TiagoEnv(robot_gazebo_env.RobotGazeboEnv):
    """
    Initializes a new Tiago Steel environment
    """
    def __init__(self):
        rospy.logdebug("========= In Tiago Env")

        #ROSLauncher(rospackage_name="tiago_gym",
        #            launch_file_name="tiago_steel_gazebo.launch",
        #            ros_ws_abspath=ros_ws_abspath)

        self.controllers_list = []
        self.robot_name_space = ""

        # Whether to reset controllers when a new episode starts
        reset_controls_bool = False
        
        # Parent class init
        super(TiagoEnv, self).__init__(controllers_list=self.controllers_list,
                                       robot_name_space=self.robot_name_space,
                                       reset_controls=reset_controls_bool,
                                       reset_world_or_sim="WORLD")
        
        self.gazebo.unpauseSim()
        self._init_moveit()
        self._init_rviz_marker()
        self._check_all_systems_ready()

        self.gazebo.pauseSim()

    # Methods needed by the RobotGazeboEnv
    # ----------------------------
    def _check_all_systems_ready(self):
        """
        Checks that all the sensors, publishers and other simulation systems are
        operational.
        """
        return True

    # TiagoEnv virtual methods
    # ----------------------------

    def _init_moveit(self):
        moveit_commander.roscpp_initialize(sys.argv)
        robot = moveit_commander.RobotCommander()
        scene = moveit_commander.PlanningSceneInterface()

        # TODO: add other groups: 'head', 'gripper'
        self.arm_group = moveit_commander.MoveGroupCommander('arm')

    def _init_rviz_marker(self):
        self.marker_publisher = rospy.Publisher('visualization_marker', Marker, queue_size=100, latch=True)
                                                 
        
    # Methods that the TrainingEnvironment will need.
    # ----------------------------

    def update_marker(self, x, y, z, ns):
        marker = Marker()
        marker.header.frame_id = '/base_link'
        marker.type = marker.SPHERE
        marker.action = marker.ADD
        marker.ns = ns
        marker.id = 0

        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.1

        if ns == 'goal':
            marker.color.g = 1.0
        elif ns == 'action':
            marker.color.b = 1.0
        marker.color.a = 0.5

        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = z - 0.3

        self.marker_publisher.publish(marker)

    def get_arm_limits(self):
        return 

    def get_arm_pose(self):
        return self.arm_group.get_current_pose().pose

    def send_arm_pose(self, x, y, z, roll, pitch, yaw):
        #pose = self.arm_group.get_current_pose().pose
        pose = Pose()

        pose.position.x = x
        pose.position.y = y
        pose.position.z = z

        #q = quaternion_from_euler(roll, pitch, yaw)
        pose.orientation.x = -0.5#q[0]
        pose.orientation.y = 0.5#q[1]
        pose.orientation.z = 0.5#q[2]
        pose.orientation.w = 0.5#q[3]

        self.arm_group.set_pose_target(pose)
        plan = self.arm_group.go(wait=True)

        self.arm_group.stop()
        self.arm_group.clear_pose_targets()

        return plan
    
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
