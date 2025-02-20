#!/usr/bin/env python3

import rospy
from std_msgs.msg import String
import roslib.packages
import os
import cv2
import time
import numpy as np

from std_msgs.msg import Header 
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

from gesture_control.msg import draw, plot, timestamps
from gesture_utils.gesture_detector import GestureDetector
from gesture_utils.gesture_detector import GestureDetector
from gesture_utils.ros_utils import convert_matrix_to_ROSpoints
from gesture_utils.fps_counter import FPS_Counter
from gesture_utils.drawing_utils import draw_on_frame
from gesture_utils.control_mode_selector import GestureInterpreter

# Find the model directory absolute path
model_realtive_path = "src/gesture_utils/training/exported_model"
package_path = roslib.packages.get_pkg_dir('gesture_control')
model_absolute_path = file_path = os.path.join(package_path, model_realtive_path)

# Define the detector object
detector = GestureDetector(
    model_absolute_path,
    0.2
)

# Initialise FPS counter
fps_counter = FPS_Counter()

# Initialise timestamp arrays
timestamps_times = []
timestamps_names = []

# Camera calibration parameters
cal_res_relative_path = "src/gesture_utils/camera_calibration/camera_calibration_results.npz"
cal_res = np.load(os.path.join(package_path, cal_res_relative_path))
camera_matrix = cal_res["camera_matrix"]
dist_coeffs = cal_res["dist_coeffs"]










def main():
    
    # Start the node
    rospy.init_node('gesture_detector', anonymous=True)
    rate = rospy.Rate(30)
    
    # Create the framework selector
    interpreter = GestureInterpreter()
    
    # Initialize the publishers
    draw_publisher = rospy.Publisher('draw_topic', Image, queue_size=1)
    plot_publisher = rospy.Publisher('plot_topic', plot, queue_size=10)
    timestamps_publisher = rospy.Publisher('timestamps_topic', timestamps, queue_size=10)
    
    # Initialize the bridge
    bridge = CvBridge()

    # Open camera
    cam_id = rospy.get_param('/detection_node/camera_id', 2)
    cam = cv2.VideoCapture(cam_id)

    rospy.loginfo(f"Camera matrix: \n{camera_matrix}; distortion coefficients: {dist_coeffs}")
    
    
    
    
    # Loop indefinitely
    while not rospy.is_shutdown():
        
        timestamps_times = []
        timestamps_names = []
        
        ############# GESTURE DETECTION #############
        
        # Capture frame
        capture_start_time = time.time()
        ret, frame = cam.read()
        #frame = cv2.undistort(frame, camera_matrix, dist_coeffs)
        if not ret: continue
        frame = cv2.flip(frame, 1)
        capture_exec_time = time.time() - capture_start_time
        timestamps_names.append('capture')
        timestamps_times.append(capture_exec_time)        
        
        fps = fps_counter.get_fps()
        
        # Process frame (detect hands and pose)
        detector.process(frame)
        timestamps_names.append('landmarks')
        timestamps_times.append(detector.landmarks_exec_time)
        timestamps_names.append('gesture')
        timestamps_times.append(detector.gesture_exec_time)
        
        # Extract gestures
        rh_gesture, lh_gesture = detector.get_hand_gestures()
        #rospy.loginfo(f"RHG:{rh_gesture}, LHG:{lh_gesture}")
        #rospy.loginfo(f"FPS: {fps}")
        
        
        ############# GESTURE INTERPRETATION #############

        ts_names, ts_values = [], []
        
        # Call the gesture interpreter
        interpret_start_time = time.time()
        interpretation_result = interpreter.interpret_gestures(
            frame=frame,
            fps=fps,
            rhg=rh_gesture,
            lhg=lh_gesture,
            rhl=detector.right_hand_landmarks_matrix,
            rhwl = detector.right_hand_world_landmarks_matrix,
            lhl=detector.left_hand_landmarks_matrix,
            pl=detector.pose_landmarks_matrix,
            camera_matrix = camera_matrix,
            dist_coeffs = dist_coeffs
        )
        interpret_exec_time = time.time() - interpret_start_time
        timestamps_names.append('interpret')
        timestamps_times.append(interpret_exec_time)

        if isinstance(interpretation_result, tuple):
            callback, ts_names, ts_values = interpretation_result
        else:
            callback = interpretation_result
        
        # Execute the selected callback        
        result = callback()
        #rospy.loginfo(f"Selected function {callback.func.__name__}()    ->     {result}")
        
        
        
        ############# DRAWING #############
        
        # Draw stuff on the frame 
        drawing_start_time = time.time()
        draw_on_frame(
            frame=frame,
            rhg=rh_gesture,
            lhg=lh_gesture,
            rhl=detector.right_hand_landmarks_matrix,
            lhl=detector.left_hand_landmarks_matrix,
            pl=detector.pose_landmarks_matrix,
            fps=fps,
            framework_names=interpreter.framework_names,
            candidate_framework=interpreter.candidate_framework_index,
            selected_framework=interpreter.selected_framework_index,
            #min_theta=interpreter.menu_manager.min_theta,
            #max_theta=interpreter.menu_manager.max_theta
        )        
        
        # Convert frame to ROS image
        ros_image = bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        
        # Publish on the draw topic
        draw_publisher.publish(ros_image)
        rospy.logdebug("Published image and landmarks to /draw_topic") 
        
        drawing_exec_time = time.time() - drawing_start_time
        timestamps_names.append('drawing')
        timestamps_times.append(drawing_exec_time)  
        
        
        
        
        ############# PLOTTING #############
        
        # Check if plotting node is listening
        if plot_publisher.get_num_connections() > 0:
        
            # Convert world coordinates to a list of ROS points
            rhwl = convert_matrix_to_ROSpoints(detector.right_hand_landmarks_matrix)
            lhwl = convert_matrix_to_ROSpoints(detector.left_hand_world_landmarks_matrix)
            pwl = convert_matrix_to_ROSpoints(detector.pose_world_landmarks_matrix)
            
            # Publish on the plot topic
            plot_msg = plot()
            plot_msg.header = Header()
            plot_msg.rh_landmarks = rhwl
            plot_msg.lh_landmarks = lhwl
            plot_msg.pose_landmarks = pwl
            plot_publisher.publish(plot_msg)
            rospy.logdebug("Published landmarks to /plot_topic") 
            
            
            
        # Publish execution times
        timestamps_msg = timestamps()
        timestamps_msg.header = Header()
        timestamps_msg.header.stamp = rospy.Time.now()
        timestamps_msg.control_mode = interpreter.selected_framework_manager.framework_name
        timestamps_msg.rhg = rh_gesture if rh_gesture is not None else 'None'
        timestamps_msg.lhg = lh_gesture if lh_gesture is not None else 'None'
        timestamps_msg.operation_names = timestamps_names + ts_names
        timestamps_msg.execution_times = timestamps_times + ts_values
        timestamps_msg.num_hands = detector.num_hands
        timestamps_publisher.publish(timestamps_msg)  
        
        rate.sleep()
        
    cam.release()






if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
    

    
    


