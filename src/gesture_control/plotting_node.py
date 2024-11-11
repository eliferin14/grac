#!/usr/bin/env python

import rospy
import numpy as np
import cv2
from cv_bridge import CvBridge

from gesture_control.msg import plot
from gesture_utils.ros_utils import convert_ROSpoints_to_matrix



bridge = CvBridge()

def plot_callback(msg):
    
    # Extract data
    rhl_ros = msg.rh_landmarks
    lhl_ros = msg.lh_landmarks
    pl_ros = msg.pose_landmarks
    
    # Convert points array to numpy matrices
    rhl = convert_ROSpoints_to_matrix(rhl_ros)
    lhl = convert_ROSpoints_to_matrix(lhl_ros)
    pl = convert_ROSpoints_to_matrix(pl_ros)
    
    print(rhl)
    
    

def listen_draw_topic():
    
    # Initialize node
    rospy.init_node("plotting")
    
    # Subscribe to the draw topic
    rospy.Subscriber('/plot_topic', plot, plot_callback)
    
    # Keep the listener spinning
    rospy.spin()
    
    
    

if __name__ == "__main__":
    try:
        listen_draw_topic()
    except rospy.ROSInterruptException:
        pass