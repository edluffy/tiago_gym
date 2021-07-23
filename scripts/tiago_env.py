import rospy
from openai_ros import robot_gazebo_env
from openai_ros.openai_ros_common import ROSLauncher
from sensor_msgs.msg import JointState

class TiagoEnv(robot_gazebo_env.RobotGazeboEnv):
    """Tiago Steel environment
    """
    def __init__(self, ros_ws_abspath):
        rospy.logdebug("========= In Tiago Env")
        ROSLauncher(rospackage_name="rl_tiago",
                    launch_file_name="tiago_steel_gazebo.launch",
                    ros_ws_abspath=ros_ws_abspath)

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
        self.controllers_object.reset_controllers()
        self._check_all_sensors_ready()

        # 'rostopic info /joint_states'
        self.joint_states_sub = rospy.Subscriber('/joint_states', JointState, self.joints_callback)
        self.joints = JointState()

    # Methods needed by the RobotGazeboEnv
    # ----------------------------
    def _check_all_systems_ready(self):
        """
        Checks that all the sensors, publishers and other simulation systems are
        operational.
        """
        self._check_all_sensors_read()
        return True

    # TiagoEnv virtual methods
    # ----------------------------

    def _check_all_sensors_ready(self):
        self._check_joint_states_ready()

        rospy.logdebug("ALL SENSORS READY")

    def _check_joint_states_ready(self):
        self.joints = None
        while self.joints is None and not rospy.is_shutdown():
            try:
                self.joints = rospy.wait_for_message

    
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
