#!/usr/bin/env python3

import rospy
import numpy as np
from functools import partial
import time
from filterpy.kalman import KalmanFilter
from scipy.signal import savgol_filter
import cv2

from gesture_utils.frameworks.cartesian_control_mode import CartesianControlMode

from moveit_msgs.srv import GetPositionIK, GetPositionIKRequest

from geometry_msgs.msg import Point






class ExponentialMovingAverage():

    def __init__(self, tau):
        """
        Initializes the filter.
        
        Args:
            tau (float): Time constant controlling the filter's smoothness.
        """
        self.tau = tau
        self.prev_filtered = None
        self.prev_time = None

    def update(self, value, current_time):
        """
        Updates the filter with a new value and timestamp.

        Args:
            value (float): The new input value.
            current_time (float): The current timestamp.

        Returns:
            float: The filtered value.
        """
        if self.prev_time is None:
            # First call: Initialize the filter with the input value
            self.prev_filtered = value
            self.prev_time = current_time
            return self.prev_filtered

        # Calculate time difference
        delta_t = current_time - self.prev_time

        # Calculate time-normalized decay factor
        alpha = 1 - np.exp(-delta_t / self.tau)

        # Update the filtered value
        self.prev_filtered = alpha * value + (1 - alpha) * self.prev_filtered

        # Update previous time
        self.prev_time = current_time

        return self.prev_filtered
    
    def reset(self):
        self.prev_time = None
        self.prev_filtered = None


class PositionFilter():

    def __init__(self, tau_x, tau_y, tau_z, tau_rx, tau_ry, tau_rz):

        # Initialise all the filters
        self.filters = [
            ExponentialMovingAverage(tau_x), 
            ExponentialMovingAverage(tau_y), 
            ExponentialMovingAverage(tau_z),
            #ExponentialMovingAverage(tau_rx), 
            #ExponentialMovingAverage(tau_ry), 
            #ExponentialMovingAverage(tau_rz)
        ]

    def update(self, new_position, current_time):
        self.filtered_value = []

        for i, f in enumerate(self.filters):
            self.filtered_value.append( f.update(new_position[i], current_time) )

        assert len(self.filtered_value) == len(new_position)

        return np.array(self.filtered_value)
    
    def reset(self):
        for f in self.filters:
            f.reset()







class ConstantAccelerationKalmanFilter:
    def __init__(self, process_noise=0.01, measurement_noise=1.0):
        self.kf = KalmanFilter(dim_x=9, dim_z=3)

        # Measurement function (H) - Only observes position
        self.kf.H = np.zeros((3, 9))
        self.kf.H[:, :3] = np.eye(3)

        # Process noise covariance (Q) - models uncertainty in acceleration
        self.kf.Q = np.eye(9) * process_noise

        # Measurement noise covariance (R) - models sensor noise
        self.kf.R = np.eye(3) * measurement_noise

        # Initial state estimate
        self.kf.x = np.zeros((9, 1))  # Start at rest

        # Initial state covariance (P)
        self.kf.P = np.eye(9) * 1.0

    def update_transition_matrix(self, dt):
        """ Updates the state transition matrix F based on the given dt """
        F = np.eye(9)
        for i in range(3):
            F[i, i+3] = dt
            F[i, i+6] = 0.5 * dt**2
            F[i+3, i+6] = dt
        self.kf.F = F

    def update(self, measurement, dt):
        """ Updates the filter with a new 3D position measurement and given dt """
        self.update_transition_matrix(dt)  # Update transition model
        
        self.kf.predict()
        self.kf.update(np.array(measurement).reshape(3, 1))
        return self.kf.x[:3].flatten()  # Return estimated position





class SavitzkyGolayTrajectorySmoother:
    def __init__(self, window_size=5, poly_order=2):
        """
        Initializes the trajectory generator.
        :param window_size: Number of past points used for filtering (must be odd).
        :param poly_order: Order of polynomial for Savitzky-Golay filter.
        """
        self.window_size = window_size
        self.poly_order = poly_order
        self.history = []  # Stores past measurements

    def update(self, measurement):
        """
        Updates the trajectory with a new noisy measurement and predicts the next position.
        :param measurement: (x, y, z) tuple
        :return: Estimated next position (x, y, z)
        """
        self.history.append(measurement)

        # If there are no measurements stored, return the raw measurement
        if len(self.history) <= self.poly_order:
            return measurement
        
        # If there are less that the desired number of points, smooth anyways but with less points
        ws = np.min([ len(self.history), self.window_size ])

        # Keep only the last N points
        self.history = self.history[-ws:]

        # Convert to NumPy array for processing
        data = np.array(self.history)

        # Apply Savitzky-Golay filter separately for x, y, z
        smoothed_x = savgol_filter(data[:, 0], ws, self.poly_order, mode='nearest')
        smoothed_y = savgol_filter(data[:, 1], ws, self.poly_order, mode='nearest')
        smoothed_z = savgol_filter(data[:, 2], ws, self.poly_order, mode='nearest')

        # Compute estimated next position by extrapolating the last trend
        next_x = 2 * smoothed_x[-1] - smoothed_x[-2]
        next_y = 2 * smoothed_y[-1] - smoothed_y[-2]
        next_z = 2 * smoothed_z[-1] - smoothed_z[-2]

        return [next_x, next_y, next_z]

    def reset(self):
        self.history = []


    







class HandMimicControlMode( CartesianControlMode ):
    
    framework_name = "Mimic hand"
    
    left_gestures_list = ['one', 'two', 'three', 'four']
    scaling_list_length = len(left_gestures_list)

    # Initialise filters
    ema_position_filter = PositionFilter(0.3,0.3,0.3,0.001,0.001,0.001)
    ca_filter = ConstantAccelerationKalmanFilter(process_noise=0.01, measurement_noise=100)
    sg_smoother = SavitzkyGolayTrajectorySmoother(window_size=9, poly_order=2)

    # Define a scaling factor for the depth coordinate
    depth_scaling = 1
    
    # TODO: Define this matrix properly
    camera_to_robot_tf = np.vstack([ [-1,0,0], [0,0,1], [0,-1,0] ])
    #camera_to_robot_tf = np.eye(3)

    camera_matrix = np.eye(3, dtype=np.float32)
    
    
    def __init__(self, group_name="manipulator", min_scaling=0.1, max_scaling=5):
        
        super().__init__(group_name)
        
        # Initialise the service caller
        self.ik_service = rospy.ServiceProxy('/compute_ik', GetPositionIK)
        
        # Declare the starting points for hand and robot
        self.hand_previous_position, self.robot_previous_position = None, None
        self.hand_previous_orientation, self.robot_previous_orientation = None, None
        self.is_mimicking = False
        
        # Calculate the scaling values
        self.scaling_list = np.logspace( np.log10(min_scaling), np.log10(max_scaling), self.scaling_list_length)
        rospy.loginfo(f"Scaling values: {self.scaling_list}")

        self.publisher_raw = rospy.Publisher('hand_position_raw', Point, queue_size=10)
        self.publisher_filtered = rospy.Publisher('hand_position_filtered', Point, queue_size=10)
        
        
        
        
        
    def interpret_gestures(self, *args, **kwargs):
        
        # Extract parameters from kwargs
        rhg = kwargs['rhg']
        lhg = kwargs['lhg']
        pl = kwargs['pl']
        rhl = kwargs['rhl']
        rhwl = kwargs['rhwl']
        
        
        
        
        
        ############## ACTION SELECTION #######################
    
        # If the lhg is not in the list, do nothing
        # Check the right hand: move only when it is in 'pick'
        if lhg not in self.left_gestures_list    or   not rhg == 'pick' :
            # Reset the flag
            self.is_mimicking = False

            # Reset all the filters
            self.ema_position_filter.reset()
            self.sg_smoother.reset()

            # Store the starting time
            self.start_time = time.time() 

            return partial(self.stop)      
        

        # Measure the time that has passed from the beginning of the mimicking (needed for filtering)
        current_time = time.time() - self.start_time   

        # Get hand depth from world coordinates and pixel coordinates
        coplanar_points_indexes = np.arange(21)
        points3D = rhwl[coplanar_points_indexes]
        points2D = rhl[:,:2][coplanar_points_indexes]
        hand_current_position = self.solvePnP(points3D, points2D)
        
        # Apply filtering to the hand position
        filtered_position = self.ema_position_filter.update(hand_current_position, current_time)
        #filtered_position = self.ca_filter.update(hand_current_position, current_time)
        #filtered_position = hand_current_position
        #rospy.loginfo(f"{filtered_position[0]:.3f}\t{filtered_position[1]:.3f}\t{filtered_position[2]:.3f}")

        # Publish the raw position
        point = Point()
        point.x = hand_current_position[0]
        point.y = hand_current_position[1]
        point.z = hand_current_position[2]
        self.publisher_raw.publish(point)

        # Publish the filtered position
        point = Point()
        point.x = filtered_position[0]
        point.y = filtered_position[1]
        point.z = filtered_position[2]
        self.publisher_filtered.publish(point)

        # Get current pose of the robot and store it
        robot_pose = self.group_commander.get_current_pose().pose
        robot_position, robot_orientation = self.convert_pose_to_p_q(robot_pose)

        
        # If the robot is not already mimicking, define the starting points and flip the flag
        if not self.is_mimicking:
            
            rospy.logdebug("Saving starting postion of robot and hand")
            
            # This code is supposed to be executed only once, when the left hand is choosing the scaling AND the right hand is 'fist'
            
            # Save the starting position of the hand
            self.hand_previous_position = hand_current_position
            self.hand_previous_orientation = None  

            self.robot_previous_position = self.sg_smoother.update(robot_position)
            self.robot_previous_orientation = robot_orientation

            # Store the starting time
            self.start_time = time.time()   
            
            # Flip the flag
            self.is_mimicking = True
            
            # Stay still
            return partial(self.stop)
            
        
        
        
        
        ############## TARGET DEFINITION ######################
            
        # Select the scaling
        index = self.left_gestures_list.index(lhg)
        scaling_factor = self.scaling_list[index]     
        
        # Calculate the delta vector between the current and starting position of the right hand
        rospy.logdebug(f"Hand starting position: {self.hand_previous_position}")
        rospy.logdebug(f"Hand current position: {filtered_position}")
        hand_delta_position = filtered_position - self.hand_previous_position
        hand_delta_position = self.suppress_noise(hand_delta_position, 5e-4)
        #rospy.loginfo(f"Hand delta position: {hand_delta_position}")

        # Compensate depth scaling
        hand_delta_position[2] *= self.depth_scaling
 
        # Apply scaling to obtain the delta vector for the robot
        robot_delta_position = scaling_factor * hand_delta_position
        rospy.logdebug(f"Robot delta position before rotation: {robot_delta_position}")
        
        # Change of coordinates to have the axes of the camera aligned with the robot base frame
        robot_delta_position = self.camera_to_robot_tf @ robot_delta_position
        #rospy.loginfo(f"Robot delta position: {robot_delta_position}")
        
        # Calculate the target pose for the robot (starting pose + scaled delta vector)
        robot_target_position = self.robot_previous_position + robot_delta_position
        robot_target_position = self.sg_smoother.update(robot_target_position)
        robot_target_orientation = self.robot_previous_orientation
        robot_target_pose = self.convert_p_q_to_pose(robot_target_position, robot_target_orientation)
        rospy.logdebug(robot_target_pose)

        # Substitute the previous position with the current one for the next iteration
        self.hand_previous_position = filtered_position      
        self.robot_previous_position = robot_target_position #NOTE This could be changed to the current robot position instead of the previous target



        
        # Call the inverse kinematics service
        target_joints = self.compute_ik(robot_target_pose)
        rospy.logdebug(f"Target joints: {target_joints}")
        
        # Generate the goal 
        goal = self.generate_action_goal(target_joints, self.joint_names)
        
        # Send the goal to the action server
        action_request = partial(self.client.send_goal, goal=goal)        
        return action_request




    def suppress_noise(self, values, threshold):

        for i, v in enumerate(values):
            if np.abs(v) < threshold:
                values[i] = 0

        return values
    



    def solvePnP(self, points3D, points2D):

        assert points2D.shape[1] == 2
        assert points3D.shape[0] == points2D.shape[0]

        """ normAB = np.linalg.norm(points3D[0] - points3D[1])
        normBC = np.linalg.norm(points3D[2] - points3D[1])
        normAC = np.linalg.norm(points3D[0] - points3D[2])

        rospy.loginfo(f"AB: {normAB:.4f}, BC: {normBC:.4f}, AC: {normAC:.4f}") """

        # Get rotation and translation vectors from points
        ret, rvec, tvec, _ = cv2.solvePnPRansac(points3D, points2D, self.camera_matrix, np.zeros((5,1)))
        #rospy.loginfo(f"Translation vector: {tvec}")
        if not ret:
            raise ValueError

        # Get the position of the wrist base in camera frame
        R, _ = cv2.Rodrigues(rvec)
        wrist_pos = R @ points3D[0].reshape((-1)) + tvec.reshape((-1))

        rospy.loginfo(f"Estimated wrist positon in camera frame: {wrist_pos}")

        return wrist_pos.reshape((-1))
            
            












if __name__ == "__main__":
    
    rospy.init_node("Hand_mimic", log_level=rospy.DEBUG)
    
    hmfm = HandMimicControlMode()
    
    lhg = 'one'
    rhg = 'fist'
    pl = np.zeros((33,3), dtype=float)
    
    hmfm.interpret_gestures(lhg=lhg, rhg=rhg, pl=pl)
    
    pl[15] = np.random.uniform(-0.5,0.5,3)
    print(pl)
    
    callback = hmfm.interpret_gestures(lhg=lhg, rhg=rhg, pl=pl)
    callback()
    
    