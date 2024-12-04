#!/usr/bin/env python

import rospy
from std_msgs.msg import String
import roslib.packages
import os
import cv2

from std_msgs.msg import Header 
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

from gesture_control.msg import draw, plot
from gesture_utils.gesture_detector import GestureDetector
from gesture_utils.gesture_detector import GestureDetector
from gesture_utils.ros_utils import convert_matrix_to_ROSpoints
from gesture_utils.fps_counter import FPS_Counter
from gesture_utils.drawing_utils import draw_on_frame

# Find the model directory absolute path
model_realtive_path = "src/gesture_utils/training/exported_model"
package_path = roslib.packages.get_pkg_dir('gesture_control')
model_absolute_path = file_path = os.path.join(package_path, model_realtive_path)

# Define the detector object
detector = GestureDetector(
    model_absolute_path,
    0.2
)

# Open camera
cam = cv2.VideoCapture(3)

# Initialise FPS counter
fps_counter = FPS_Counter()







# Create arm object
# Don't forget to launch the robot simulator!
from sami.arm import Arm, EzPose
arm = Arm('ur10e_moveit', group='manipulator')

# Create the framework selector
from gesture_utils.framework_selector import FrameworkSelector










def gesture_detection():
    
    # Start the node
    rospy.init_node('gesture_detector', anonymous=True)
    rate = rospy.Rate(30)
    
    # Create the framework selector
    interpreter = FrameworkSelector()
    
    # Initialize the publishers
    draw_publisher = rospy.Publisher('draw_topic', draw, queue_size=1)
    plot_publisher = rospy.Publisher('plot_topic', plot, queue_size=10)
    
    # Initialize the bridge
    bridge = CvBridge()
    
    
    
    
    # Loop indefinitely
    while not rospy.is_shutdown():
        
        ############# GESTURE DETECTION #############
        
        # Capture frame
        ret, frame = cam.read()
        if not ret: continue
        frame = cv2.flip(frame, 1)
        
        fps = fps_counter.get_fps()
        
        # Process frame (detect hands and pose)
        detector.process(frame)
        
        # Extract gestures
        rh_gesture, lh_gesture = detector.get_hand_gestures()
        #rospy.loginfo(f"FPS: {fps}")
        
        
        ############# GESTURE INTERPRETATION #############
        
        # Call the gesture interpreter
        callback = interpreter.interpret_gestures(
            frame=frame,
            fps=fps,
            arm=arm,
            rhg=rh_gesture,
            lhg=lh_gesture,
            rhl=detector.right_hand_landmarks_matrix,
            lhl=detector.left_hand_landmarks_matrix
        )
        
        # Execute the selected callback        
        result = callback()
        #rospy.loginfo(f"Selected function {callback.func.__name__}()    ->     {result}")
        
        
        
        ############# DRAWING #############
        
        # Check if the drawing node is listening
        if draw_publisher.get_num_connections() > 0:
        
            # Draw stuff on the frame 
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
                min_theta=interpreter.menu_manager.min_theta,
                max_theta=interpreter.menu_manager.max_theta
            )        
            
            # Convert frame to ROS image
            ros_image = bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            
            # Publish on the draw topic
            draw_msg = draw()
            draw_msg.header = Header()
            draw_msg.frame = ros_image
            draw_publisher.publish(draw_msg)
            rospy.logdebug("Published image and landmarks to /draw_topic")   
        
        
        
        
        ############# PLOTTING #############
        
        # Check if plotting node is listening
        if plot_publisher.get_num_connections() > 0:
        
            # Convert world coordinates to a list of ROS points
            rhwl = convert_matrix_to_ROSpoints(detector.right_hand_world_landmarks_matrix)
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
        
        rate.sleep()
        
    cam.release()






if __name__ == '__main__':
    try:
        gesture_detection()
    except rospy.ROSInterruptException:
        pass
    

    
    


