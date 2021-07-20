#! /usr/bin/env python
import rospy
import actionlib

from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal, FollowJointTrajectoryFeedback, FollowJointTrajectoryResult
from trajectory_msgs.msg import JointTrajectoryPoint


rospy.init_node('trajectory_controller')
client = actionlib.SimpleActionClient('arm_controller/follow_joint_trajectory', FollowJointTrajectoryAction)
client.wait_for_server()

goal = FollowJointTrajectoryGoal()
goal.trajectory.joint_names = [ 'arm_1_joint',
                                'arm_2_joint',
                                'arm_3_joint',
                                'arm_4_joint',
                                'arm_5_joint',
                                'arm_6_joint',
                                'arm_7_joint' ]

goal.trajectory.points = [JointTrajectoryPoint() for _ in range(2)]

goal.trajectory.points[0].positions = [0.2, 0.0, -1.5, 1.94, -1.57, -0.5, 0.0]
goal.trajectory.points[0].velocities = [1.0 for _ in range(7)]
goal.trajectory.points[0].time_from_start = rospy.Duration(2.0)

goal.trajectory.points[1].positions = [2.5, 0.2, -2.1, 1.9, 1.0, -0.5, 0.0]
goal.trajectory.points[1].velocities = [0.0 for _ in range(7)]
goal.trajectory.points[1].time_from_start = rospy.Duration(4.0)

goal.trajectory.header.stamp = rospy.Time.now() + rospy.Duration(1.0)


client.send_goal_and_wait(goal)
