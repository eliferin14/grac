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
from gesture_utils.ros_utils import convert_matrix_to_ROSpoints
from gesture_utils.fps_counter import FPS_Counter

# Find the model directory absolute path
model_realtive_path = "gesture_utils/training/exported_model"
package_path = roslib.packages.get_pkg_dir('gesture_control')
model_absolute_path = file_path = os.path.join(package_path, model_realtive_path)

# Define the detector object
detector = GestureDetector(
    model_absolute_path,
    0.3
)

# Open camera
cam = cv2.VideoCapture(3)

fps_counter = FPS_Counter()

def talker():
    
    # Start the node
    rospy.init_node('gesture_detector', anonymous=True)
    rate = rospy.Rate(50)  # 10hz
    
    # Initialize the publishers
    draw_publisher = rospy.Publisher('draw_topic', draw, queue_size=1)
    plot_publisher = rospy.Publisher('plot_topic', plot, queue_size=10)
    
    # Initialize the bridge
    bridge = CvBridge()
    
    while not rospy.is_shutdown():
        
        # Capture frame
        ret, frame = cam.read()
        if not ret: continue
        frame = cv2.flip(frame, 1)
        
        # Process frame (detect hands and pose)
        detector.process(frame)
        
        # Extract gestures
        rh_gesture, lh_gesture = detector.get_hand_gestures()
        rospy.loginfo(rh_gesture)
        
        # Convert normalized image coordiantes matrix to a list of ROS points
        rhl = convert_matrix_to_ROSpoints(detector.right_hand_landmarks_matrix)
        lhl = convert_matrix_to_ROSpoints(detector.left_hand_landmarks_matrix)
        pl = convert_matrix_to_ROSpoints(detector.pose_landmarks_matrix)
        
        # Convert frame to ROS image
        ros_image = bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        
        # Publish on the draw topic
        draw_msg = draw()
        draw_msg.header = Header()
        draw_msg.frame = ros_image
        draw_msg.rh_2D_landmarks = rhl
        draw_msg.lh_2D_landmarks = lhl
        draw_msg.pose_2D_landmarks = pl
        draw_msg.rh_gesture = rh_gesture if rh_gesture is not None else ''
        draw_msg.lh_gesture = lh_gesture if lh_gesture is not None else ''
        draw_msg.fps = fps_counter.get_fps()
        draw_publisher.publish(draw_msg)
        rospy.logdebug("Published image and landmarks to /draw_topic")        
        
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
        talker()
    except rospy.ROSInterruptException:
        pass
    

    
    


