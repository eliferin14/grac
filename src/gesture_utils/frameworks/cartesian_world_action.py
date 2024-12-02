import rospy
import numpy as np
from functools import partial

from sami.arm import Arm
from gesture_utils.frameworks.base_framework import BaseFrameworkManager

import actionlib
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from moveit_msgs.srv import GetPositionIK, GetPositionIKRequest

import tf2_ros
from tf.transformations import quaternion_multiply, quaternion_about_axis, quaternion_matrix




class CartesianActionFrameworkManager(BaseFrameworkManager):
    
    framework_name = "Cartesian control (Action)"
    
    # List that defines how joints are selected
    gesture_to_dof_list = np.array(['fist', 'one', 'two', 'three', 'four', 'palm'])
    
    # To be substituted with a value from the parameter server
    angle_step = np.pi / 64
    position_step = 0.05
    time_step = 0.5
    
    
    
    
    def __init__(self, arm=Arm('ur10e_moveit', group='manipulator')):
        
            
        # Client for the action server
        self.client = actionlib.SimpleActionClient('/arm_controller/follow_joint_trajectory', FollowJointTrajectoryAction)
        #   self.client.wait_for_server(timeout=3)
    
        # Get joint names and limits from the arm object
        self.joint_names = arm.arm_interface.moveg.get_active_joints()
        self.joint_limits = CartesianActionFrameworkManager.get_joint_limits(arm.arm_interface.robot, self.joint_names)
        
        # Initialise variables
        self.old_rhg = None
        self.selected_dof_index = None
        
        # Define the reference frame
        self.reference_frame = arm.arm_interface.moveg.get_planning_frame()
    
    
    def interpret_gestures(self, *args, **kwargs):
        
        
        
    
        # Extract parameters from kwargs
        arm = kwargs['arm']
        rhg = kwargs['rhg']
        lhg = kwargs['lhg']
        
        
        
        
        
        
        ################ ACTION SELECTION ########################
        
        # The left hand selects the joint
        candidate_selected_dof = np.where(self.gesture_to_dof_list == lhg)[0]
        
        # If no joint is selected (left hand is not in any of the fist->palm gestures), do nothing
        if candidate_selected_dof.size == 0:
            return partial(super().dummy_callback)
        
        # Change selected joint
        if candidate_selected_dof != self.selected_dof_index: rospy.loginfo(f"Joint {candidate_selected_dof[0]} selected")
        self.selected_dof_index = candidate_selected_dof[0]
        
        # The right hand selects the angle
        angle_step = 0
        position_step = 0
        if rhg == 'one': 
            angle_step = self.angle_step         # positive movement
            position_step = self.position_step
        elif rhg == 'two': 
            angle_step = -self.angle_step      # negative movement
            position_step = -self.position_step
        else: return partial(super().dummy_callback)
        
        
        
        
        
            
            
            
    
        ################# INVERSE KINEMATICS #######################
        
        # Get current pose
        current_pose = arm.get_pose()
        pose_target = current_pose
        current_orientation = pose_target.orientation
        
        # Get quaternion of the current pose
        q_current = [current_orientation.x, current_orientation.y, current_orientation.z, current_orientation.w]
        
        # Define the rotation matrix:
        # World frame: [ [1,0,0], [0,1,0], [0,0,1] ]
        # EE frame: quaternion_matrix(quaternion)
        # Then select the columns: R = [ x,y,z ]
        
        # Initialise quaternions to calculate the rotation
        q_final = q_current
        q_rotation = [0,0,0,1]
        
        # Based on the selected joint, do stuff
        if self.selected_dof_index < 3 and self.selected_dof_index >= 0:        # Linear movement
            
            if self.selected_dof_index == 0:
                pose_target.position.x += position_step
            if self.selected_dof_index == 1:
                pose_target.position.y += position_step
            if self.selected_dof_index == 2:
                pose_target.position.z += position_step
            
        elif self.selected_dof_index < 6 and self.selected_dof_index >= 3:
            
            # Select rotation axis
            if self.selected_dof_index == 3:                                        # Roll
                rotation_axis = [1,0,0]
            elif self.selected_dof_index == 4:                                      # Pitch
                rotation_axis = [0,1,0]
            elif self.selected_dof_index == 5:                                      # Yaw
                rotation_axis = [0,0,1]
                
            # Calculate the rotation quaternion and final quaternion
            q_rotation = quaternion_about_axis(angle_step, rotation_axis)
            q_final = quaternion_multiply(q_rotation, q_current)
                
        else:
            rospy.logerr("Invalid DoF selected")
            
        # Build the target
        pose_target.orientation.x = q_final[0]
        pose_target.orientation.y = q_final[1]
        pose_target.orientation.z = q_final[2]
        pose_target.orientation.w = q_final[3]
        
        # Call the ROS service for inverse kinematics
        compute_ik = rospy.ServiceProxy('/compute_ik', GetPositionIK)
        request = GetPositionIKRequest()
        request.ik_request.group_name = "manipulator"
        request.ik_request.pose_stamped.header.frame_id = "base_link"
        request.ik_request.pose_stamped.pose = pose_target
        
        response = compute_ik(request)
        
        # Compute inverse kinematics
        target_joints = response.solution.joint_state.position[:6]
        
        
        
        
        
        
        
        ################# GOAL DEFINITION ######################
        
        # Define the trajectory point
        point = JointTrajectoryPoint()
        point.positions = target_joints
        point.time_from_start = rospy.Duration(self.time_step)
        
        # Define the trajectory object
        trajectory = JointTrajectory()
        trajectory.joint_names = self.joint_names
        trajectory.points = [point]

        # Send the initial goal with no points
        goal = FollowJointTrajectoryGoal()
        goal.trajectory = trajectory
        
        
        
        
        # Return the call to the action server
        return partial(self.client.send_goal, goal=goal)
    
    
    
    
    
    # You can call this function with JointActionFrameworkManager.get_joint_limits()
    def get_joint_limits(robot_commander, joint_names):
        
        joint_limits = []
        for joint in joint_names:
            #print(robot_commander.get_joint(joint).bounds())
            joint_limits.append( robot_commander.get_joint(joint).bounds() )
            
        return joint_limits
    
    
    

    def get_frame_coordinates(frame_name, reference_frame="base_link"):
        tf_buffer = tf2_ros.Buffer()
        tf_listener = tf2_ros.TransformListener(tf_buffer)
        try:
            # Wait and get the transform
            trans = tf_buffer.lookup_transform(reference_frame, frame_name, rospy.Time(0), rospy.Duration(1.0))
            position = trans.transform.translation
            orientation = trans.transform.rotation
            return position, orientation
        except tf2_ros.LookupException as e:
            rospy.logerr(f"Frame '{frame_name}' not found: {e}")
            return None, None
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
if __name__ == "__main__":
    manager = CartesianActionFrameworkManager()