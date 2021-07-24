#! /usr/bin/env python
import rospy
import actionlib
from openai_ros import robot_gazebo_env
from openai_ros.openai_ros_common import ROSLauncher
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal, JointTrajectoryControllerState
from trajectory_msgs.msg import JointTrajectoryPoint

class TiagoEnv(robot_gazebo_env.RobotGazeboEnv):
    """
    Initializes a new Tiago Steel environment
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
        reset_controls_bool = True
        
        # Parent class init
        super(TiagoEnv, self).__init__(controllers_list=self.controllers_list,
                                       robot_name_space=self.robot_name_space,
                                       reset_controls=reset_controls_bool)
        
        self.gazebo.unpauseSim()
        self._check_all_systems_ready()

        # Subscribe to sensor topics
        rospy.Subscriber('/arm_controller/state', JointTrajectoryControllerState, self._arm_state_callback)
        # TODO: add head and gripper sensors

        # Initialize trajectory controller
        self.controller_object = TiagoController()

        self.gazebo.pauseSim()

    # Methods needed by the RobotGazeboEnv
    # ----------------------------
    def _check_all_systems_ready(self):
        """
        Checks that all the sensors, publishers and other simulation systems are
        operational.
        """
        self._check_arm_state_ready()
        rospy.logdebug("ALL SENSORS READY")
        return True

    # TiagoEnv virtual methods
    # ----------------------------

    def _check_arm_state_ready(self):
        self.arm_state = None
        rospy.logdebug( "Waiting for /arm_controller/state to be READY...")
        while self.arm_state is None and not rospy.is_shutdown():
            try:
                self.arm_state = rospy.wait_for_message(
                    "/arm_controller/state", JointTrajectoryControllerState, timeout=5.0)
                rospy.logdebug("Current /arm_controller/state READY=>")
            except:
                rospy.logerr("Current /arm_controller/state not ready yet")

    def _arm_state_callback(self, msg):
        self.arm_state = msg
    
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

    def get_joint_limits(self):
        return self.controller_object.arm_joints_dict.values()

    def get_joint_positions(self):
        return self.arm_state.actual.positions

    def set_joint_positions(self, positions):
        assert len(positions) == 7, 'Wrong number of joints given'
        self.controller_object.send_arm_trajectory([positions])

class TiagoController(object):
    def __init__(self):

        # TODO: add other clients : '/head_controller', '/gripper_controller'

        # joint_name: [upper_limit, lower_limit]
        self.arm_joints_dict = {'arm_1_joint': (0, 0),
                                'arm_2_joint': (0, 0),
                                'arm_3_joint': (0, 0),
                                'arm_4_joint': (0, 0),
                                'arm_5_joint': (0, 0),
                                'arm_6_joint': (0, 0),
                                'arm_7_joint': (0, 0)}

        self.arm_client = actionlib.SimpleActionClient('arm_controller/follow_joint_trajectory', FollowJointTrajectoryAction)
        self.arm_goal = FollowJointTrajectoryGoal()
        self.arm_goal.trajectory.joint_names = self.arm_joints_dict.keys()
        self.arm_client.wait_for_server()

    def get_arm_feedback(self, feedback):
        rospy.loginfo("##### ARM FEEDBACK ######")
        rospy.loginfo(str(feedback.error.positions))
        rospy.loginfo("##### ###### ######")
    
    def send_arm_trajectory(self, positions_array):
        self.arm_goal.trajectory.points = [JointTrajectoryPoint() for _ in range(len(positions_array))]

        for point in self.arm_goal.trajectory.points:
            point.positions = positions_array.pop(0)
            point.velocities = [0.0 for _ in range(7)]
            point.time_from_start = rospy.Duration(2.0)

        self.arm_goal.trajectory.header.stamp = rospy.Time.now()
        self.arm_client.send_goal(self.arm_goal, feedback_cb=self.get_arm_feedback)
