#! /usr/bin/env python
import sys
import rospy
import actionlib

from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal, FollowJointTrajectoryFeedback, FollowJointTrajectoryResult
from trajectory_msgs.msg import JointTrajectoryPoint

def move(positions):
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

    goal.trajectory.points = [JointTrajectoryPoint() for _ in range(1)]

    goal.trajectory.points[0].positions = positions
    goal.trajectory.points[0].velocities = [0.0 for _ in range(7)]
    goal.trajectory.points[0].time_from_start = rospy.Duration(2.0)

    goal.trajectory.header.stamp = rospy.Time.now() + rospy.Duration(1.0)

    client.send_goal_and_wait(goal)

if __name__=="__main__":
    sys.argv = sys.argv[1:]
    if len(sys.argv) == 7:
        move(positions=[float(p) for p in sys.argv])
    else:
        print('wrong number of joint positions:', len(sys.argv))
