#!/usr/bin/env python

import rospy
import numpy as np
import cv2
from cv_bridge import CvBridge

from gesture_control.msg import draw
from gesture_utils.ros_utils import convert_ROSpoints_to_matrix
from gesture_utils.drawing_utils import draw_hand, draw_pose, denormalize_landmarks




bridge = CvBridge()

def draw_callback(msg):
    
    # Extract data
    ros_image = msg.frame
    rhl_ros = msg.rh_2D_landmarks
    lhl_ros = msg.lh_2D_landmarks
    pl_ros = msg.pose_2D_landmarks
    
    # Convert ROS image to opencv image
    frame = bridge.imgmsg_to_cv2(ros_image, desired_encoding='bgr8')
    
    # Convert points array to numpy matrices
    rhl = convert_ROSpoints_to_matrix(rhl_ros)
    lhl = convert_ROSpoints_to_matrix(lhl_ros)
    pl = convert_ROSpoints_to_matrix(pl_ros)
    #print(rhl)
    
    # De-normalize landmarks
    height, width = frame.shape[0], frame.shape[1]
    rhl_pixel = denormalize_landmarks(rhl, width, height)
    lhl_pixel = denormalize_landmarks(lhl, width, height)
    pl_pixel = denormalize_landmarks(pl, width, height)
    #print(lhl_pixel)
    
    # Draw pose
    draw_pose(frame, pl_pixel, point_color=(0,255,0), line_color=(128,128,128))
    
    # Draw hands
    draw_hand(frame, rhl_pixel, point_color=(255,0,0), line_color=(255,255,255))
    draw_hand(frame, lhl_pixel, point_color=(0,0,255), line_color=(255,255,255))
    
    # Add text
    cv2.putText(frame, f"FPS: {msg.fps:.1f}", (50,50), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,255,0), thickness=2)
    cv2.putText(frame, f"Left: {msg.lh_gesture}", (50,100), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,0,255), thickness=2)
    cv2.putText(frame, f"Right: {msg.rh_gesture}", (50,150), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255,0,0), thickness=2)
    
    # Show the image
    cv2.imshow("Live feed", frame)
    cv2.waitKey(1)
    
    

def listen_draw_topic():
    
    # Initialize node
    rospy.init_node("drawing")
    
    # Subscribe to the draw topic
    rospy.Subscriber('/draw_topic', draw, draw_callback)
    
    # Keep the listener spinning
    rospy.spin()
    
    
    

if __name__ == "__main__":
    try:
        listen_draw_topic()
    except rospy.ROSInterruptException:
        pass