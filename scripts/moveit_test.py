#! /usr/bin/env python
import sys
import numpy as np
import rospy
import moveit_commander
import geometry_msgs.msg
from tf.transformations import quaternion_from_euler

moveit_commander.roscpp_initialize(sys.argv)
rospy.init_node('moveit_demo')

robot = moveit_commander.RobotCommander()
scene = moveit_commander.PlanningSceneInterface()

arm_group = moveit_commander.MoveGroupCommander('arm')

def move_in_3d_plane(x, y, z):
    pose = geometry_msgs.msg.Pose()
    #pose = arm_group.get_current_pose().pose

    pose.position.x = x
    pose.position.y = y
    pose.position.z = z

    roll = 0
    pitch = np.radians(90)
    yaw = np.radians(90)

    q = quaternion_from_euler(roll, pitch, yaw)
    pose.orientation.x = q[0]
    pose.orientation.y = q[1]
    pose.orientation.z = q[2]
    pose.orientation.w = q[3]

    arm_group.set_pose_target(pose)

    plan = arm_group.go(wait=True)

    if not plan:
        print('motion planner failed to find a path')
    else:
        print('motion planner success!')

    arm_group.stop()
    arm_group.clear_pose_targets()

for _ in range(10):
    x = np.random.uniform(0, 1)
    y = np.random.uniform(-0.5, 0.5)
    print('got here')
    move_in_3d_plane(x, y, 0.7)
