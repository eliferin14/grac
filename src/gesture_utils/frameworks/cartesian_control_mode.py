#!/usr/bin/env python

import rospy
import numpy as np
from functools import partial

from gesture_utils.frameworks.action_based_control_mode import ActionBasedControlMode

from moveit_msgs.srv import GetPositionIK, GetPositionIKRequest
from geometry_msgs.msg import Pose

import tf2_ros
from tf.transformations import quaternion_multiply, quaternion_about_axis, quaternion_matrix, quaternion_from_matrix











class CartesianControlMode(ActionBasedControlMode):
    
    framework_name = "Cartesian control (world)"
    
    selected_dof_index = None
    
    # The rotation matrix is used to extract the rotation axes for rotational movements
    # In the case of movements with world axes, the rotation matrix is the identity matrix
    orientation_matrix = np.eye(3)
    
    
    
    def __init__(self, group_name="manipulator", use_ee_frame=False):
        
        super().__init__(group_name)
        
        # change max scaling
        self.max_velocity_scaling = 0.1
        
        # This flag indicates if the relative motion is done wrt to world frame or end effector frame
        self.use_ee_frame = use_ee_frame
        if self.use_ee_frame:
            self.framework_name = "Cartesian control (end effector)"
        
        # Initialise the service caller
        self.ik_service = rospy.ServiceProxy('/compute_ik', GetPositionIK)
        
        
        
        
        
    def convert_pose_to_p_q(self, pose):
        """Converts a geometry_msg/Pose object to two arrays containing the position vector and the quaternion
        """   
        p = np.array([ pose.position.x, pose.position.y, pose.position.z ])
        q = np.array([ pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])
        return p, q
    
    def convert_p_q_to_pose(self, p,q):
        """Converts a position vector and a quaternion to a Pose object
        """
        pose = Pose()
        pose.position.x = p[0]
        pose.position.y = p[1]
        pose.position.z = p[2]
        pose.orientation.x = q[0]
        pose.orientation.y = q[1]
        pose.orientation.z = q[2]
        pose.orientation.w = q[3]
        
        return pose
        
        
        
        
        
    def compute_ik(self, pose_target):
        """Given the pose target, sends a request to the compute_ik service and returns the joint target
        """
        
        # Call the ROS service for inverse kinematics
        request = GetPositionIKRequest()
        request.ik_request.group_name = "manipulator"
        request.ik_request.pose_stamped.header.frame_id = "base_link"
        request.ik_request.pose_stamped.pose = pose_target
        #print(request)
        
        response = self.ik_service(request)
        
        # Check if a response was given
        # http://docs.ros.org/en/hydro/api/ric_mc/html/MoveItErrorCodes_8h_source.html
        if response.error_code.val != 1:
            rospy.logwarn("IK calculation failed")
            if response.error_code.val == response.error_code.NO_IK_SOLUTION:
                rospy.logwarn(f"NO_IK_SOLUTION")
            else:
                rospy.logwarn(f"IK generic error: {response.error_code.val}")
            return None
                
        # Extract the joint positions and return
        return response.solution.joint_state.position[:6]
            
    
    
    
    def interpret_gestures(self, *args, **kwargs):
        
        # Extract parameters from kwargs
        rhg = kwargs['rhg']
        lhg = kwargs['lhg']
        
        
        
        
        
        
        ################ ACTION SELECTION ########################
        
        # The left hand selects the joint
        candidate_selected_dof = np.where(self.left_gestures_list == lhg)[0]
        
        # If no joint is selected (left hand is not in any of the fist->palm gestures), do nothing
        if candidate_selected_dof.size == 0:
            return partial(self.stop)
        
        # Change selected joint
        if candidate_selected_dof != self.selected_dof_index: rospy.loginfo(f"DoF {candidate_selected_dof[0]} selected")
        self.selected_dof_index = candidate_selected_dof[0]
        
        # If the right hand is not commanding a move, stop immediately
        if rhg != 'one' and rhg != 'two': 
            return partial(self.stop)
        
        
        
        
        
        
        
        ################# INVERSE KINEMATICS #######################
        
        # Get current pose
        current_pose = self.group_commander.get_current_pose().pose
        pose_target = current_pose
        current_position = current_pose.position
        current_orientation = current_pose.orientation
        rospy.logdebug(f"Current pose: {current_pose}")
        
        # Get position vector and quaternion from the current pose
        p_current, o_current = self.convert_pose_to_p_q(current_pose)
        
        # If the end effector frame is selected, update the rotation matrix (if not selected, the rotation matrix is the identity)
        if self.use_ee_frame:
            self.orientation_matrix = quaternion_matrix(o_current)[:3, :3]
        
        # Initialise vectors to calculate the translation
        p_final = p_current
        
        # Initialise quaternions to calculate the rotation
        o_final = o_current
        
        # Calculate velocity scaling factor (function of hands distance)
        velocity_scaling = self.get_velocity_scaling(kwargs['lhl'], kwargs['rhl'])
        velocity_scaling = self.get_velocity_scaling(kwargs['lhl'], kwargs['rhl'], mapping='exponential', a=0.2, b=0.9, c=0.01, d=1)
        
        # Get the Jacobian of the current configuration
        current_joint = self.group_commander.get_current_joint_values()
        jacobian = self.group_commander.get_jacobian_matrix(current_joint)
        
        # Calculate the maximum velocity for each DoF in the selected frame
        # If in base frame, the orientation matrix is the identity
        cartesian_velocity_limits_base_frame = jacobian @ self.joint_velocity_limits
        cartesian_velocity_limits = np.hstack([ 
                                               self.orientation_matrix @ cartesian_velocity_limits_base_frame[:3],  # R*v_max
                                               self.orientation_matrix @ cartesian_velocity_limits_base_frame[3:6]  # R*omega_max
                                               ]) 
        assert cartesian_velocity_limits.shape == (6,)
        
        # Calculate the maximum step for each DoF in the selected frame
        delta_p_max = self.max_velocity_scaling * cartesian_velocity_limits * self.time_step
        
        # Isolate the selected DoF and scale the step according to the scaling factor and choose the direction
        direction = 1 if rhg == 'two' else -1
        delta_p = np.zeros((6))
        delta_p[self.selected_dof_index] = direction * velocity_scaling * delta_p_max[self.selected_dof_index]
        
        rospy.logdebug(f"DoF: {self.selected_dof_index}")
        
        # Based on the selected joint translate or rotate
        if self.selected_dof_index < 3 and self.selected_dof_index >= 0:        # Linear movement
            
            # Select the translation axis (x, y or z) from the relative orientation matrix
            translation_axis = self.orientation_matrix[:, self.selected_dof_index]     
            
            # Calculate the maximum velocity along the selected axis
            cartesian_velocity_limit_along_axis = translation_axis @ delta_p_max[:3]       
            rospy.loginfo(cartesian_velocity_limit_along_axis)
            
            # Calculate the translation vector in the selected frame
            delta_p_translation = direction * velocity_scaling * cartesian_velocity_limit_along_axis * translation_axis
            
            p_final = p_current + delta_p_translation
            
            rospy.loginfo(f"Current position: {p_current}, Translation axis: {translation_axis}, Maximum vector: {delta_p_max}, vector: {delta_p_translation}, target position: {p_final}")
            
        elif self.selected_dof_index < 6 and self.selected_dof_index >= 3:
            
            # Select the rotation axis from the relative orientation matrix
            rotation_axis = self.orientation_matrix[:, self.selected_dof_index - 3 ]       
            
            # Calculate the maximum velocity along the selected axis
            cartesian_velocity_limit_along_axis = rotation_axis @ delta_p_max[3:6]       
            rospy.loginfo(cartesian_velocity_limit_along_axis)
            
            # Calculate the rotation vector in the selected frame
            delta_o = direction * velocity_scaling * cartesian_velocity_limit_along_axis
                
            # Calculate the rotation quaternion and final quaternion
            q_rotation = quaternion_about_axis(delta_o, rotation_axis)
            o_final = quaternion_multiply(q_rotation, o_current)
            
            rospy.loginfo(f"Current orientation: {o_current}, rotation axis: {rotation_axis}, rot quaternion: {q_rotation}, target orientation: {o_final}")
                
        else:
            rospy.logerr("Invalid DoF selected")
            
        rospy.logdebug(f"Target position [array]: {p_final}")
        rospy.logdebug(f"Target position [array]: x={p_final[0]}, y={p_final[1]}, z={p_final[2]},")
            
        # Build the pose target object
        pose_target = self.convert_p_q_to_pose(p_final, o_final)
        
        rospy.logdebug(f"Target pose: {pose_target}")
        
        # Compute inverse kinematics
        target_joints = self.compute_ik(pose_target)     
        if target_joints is None: 
            return partial(self.stop)   
        
        
        
        
        
        ################ SEND ACTION REQUEST ####################        
        
        goal = self.generate_action_goal(target_joints, self.joint_names)
        action_request = partial(self.client.send_goal, goal=goal)        
        return action_request




    
    
    
    
    
    
    
if __name__ == "__main__":
    
    rospy.init_node("cartesian_world_node", log_level=rospy.DEBUG)
    
    manager = CartesianControlMode(use_ee_frame=True)
    
    callback = manager.interpret_gestures(lhg='fist', rhg='two')
    print(callback)
    result = callback()
    print(result)
    
    """ callback = manager.interpret_gestures(lhg='four', rhg='one')
    print(callback)
    callback() """