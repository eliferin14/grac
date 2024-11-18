#!/usr/bin/env python

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