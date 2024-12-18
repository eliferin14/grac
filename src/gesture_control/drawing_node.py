#!/usr/bin/env python3

import rospy
import numpy as np
import cv2
from cv_bridge import CvBridge

from gesture_control.msg import draw


bridge = CvBridge()

def draw_callback(msg):
    
    # Extract data
    ros_image = msg.frame
    
    # Convert ROS image to opencv image
    frame = bridge.imgmsg_to_cv2(ros_image, desired_encoding='bgr8')
    
    # Show the image
    cv2.namedWindow("Output", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Output", 960, 720)
    cv2.imshow("Output", frame)
    
    # If 'q' is pressed, kill the node
    if cv2.waitKey(1) & 0xFF == ord('q'):
        rospy.signal_shutdown("User requested shutdown.")
    
    

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