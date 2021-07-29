#! /usr/bin/env python
import sys
import rospy
import moveit_commander
import numpy as np
from openai_ros import robot_gazebo_env
from openai_ros.openai_ros_common import ROSLauncher
from tf.transformations import quaternion_from_euler
from geometry_msgs.msg import Pose, Point
from visualization_msgs.msg import Marker
from sensor_msgs.msg import JointState

class TiagoEnv(robot_gazebo_env.RobotGazeboEnv):
    """
    Initializes a new Tiago Steel environment
    """
    def __init__(self):
        rospy.logdebug("========= In Tiago Env")

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

    # TiagoEnv virtual methods
    # ----------------------------

    def _init_moveit(self):
        moveit_commander.roscpp_initialize(sys.argv)
        robot = moveit_commander.RobotCommander()
        scene = moveit_commander.PlanningSceneInterface()

        # TODO: add other groups: 'head', 'gripper'
        self.arm_group = moveit_commander.MoveGroupCommander('arm')
        self.arm_pose = self.arm_group.get_current_pose().pose

    def _init_rviz_marker(self):
        self.marker_publisher = rospy.Publisher('visualization_marker', Marker, queue_size=100, latch=True)

    # Methods that the TrainingEnvironment will need.
    # ----------------------------

    def set_marker_points(self, points, ns=''):
        marker = Marker()
        marker.header.frame_id = '/base_link'
        marker.type = marker.SPHERE_LIST
        marker.action = marker.ADD
        marker.ns = ns
        marker.id = 0

        if ns == 'goal':
            marker.color.g = 1.0
        elif ns == 'action':
            marker.color.b = 1.0

        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.1

        for x, y, z in points:
            marker.points.append(Point(x, y, z-0.3))

        marker.color.a = 0.5

        self.marker_publisher.publish(marker)

    def send_arm_pose(self, x, y, z, roll, pitch, yaw):
        self.arm_group.set_pose_target([x, y, z, roll, pitch, yaw])

        plan = self.arm_group.go(wait=True)
        self.arm_group.stop()
        self.arm_group.clear_pose_targets()
        return plan
    
    def shift_arm_pose(self, delta, dim):
        dim_list = ['x', 'y', 'z', 'roll', 'pitch', 'yaw']

        self.arm_group.shift_pose_target(value=delta, axis=dim_list.index(dim))
        plan = self.arm_group.go(wait=True)
        self.arm_group.stop()
        self.arm_group.clear_pose_targets()
        return plan

    def get_arm_pose(self):
        self.gazebo.unpauseSim()
        pose = self.arm_group.get_current_pose().pose
        x = pose.position.x
        y = pose.position.y
        z = pose.position.z

        roll, pitch, yaw = self.arm_group.get_current_rpy()
        self.gazebo.pauseSim()

        return x, y, z, roll, pitch, yaw
    
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
