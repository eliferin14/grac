#! /usr/bin/env python

import rospy
import actionlib

from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint




joint_names = [
      "shoulder_pan_joint",
      "shoulder_lift_joint",
      "elbow_joint",
      "wrist_1_joint",
      "wrist_2_joint",
      "wrist_3_joint"
]





def send_trajectory_goal():
    # Initialize the action client
    client = actionlib.SimpleActionClient('/arm_controller/follow_joint_trajectory', FollowJointTrajectoryAction)

    # Wait for the action server to start
    rospy.loginfo("Waiting for action server to start...")
    client.wait_for_server()

    # Create the goal message
    goal = FollowJointTrajectoryGoal()

    # Define the trajectory
    trajectory = JointTrajectory()
    trajectory.joint_names = joint_names

    # Define a trajectory point
    point = JointTrajectoryPoint()
    point.positions = [1.0, 0.5, 0.0]
    point.velocities = [0.0, 0.0, 0.0]
    point.time_from_start = rospy.Duration(2.0)

    # Add the point to the trajectory
    trajectory.points.append(point)

    # Assign the trajectory to the goal
    goal.trajectory = trajectory

    # Send the goal to the action server
    rospy.loginfo("Sending trajectory goal...")
    client.send_goal(goal)

    # Wait for the result
    client.wait_for_result()

    # Get the result of the execution
    result = client.get_result()
    rospy.loginfo(f"Action result: {result}")
    
    
    
    
    

if __name__ == '__main__':
    try:
        rospy.init_node('test_action_client')
        send_trajectory_goal()
    except rospy.ROSInterruptException:
        pass