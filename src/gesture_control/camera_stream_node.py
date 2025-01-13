#!/usr/bin/env python3

import rospy
from std_msgs.msg import String
import roslib.packages
import os
import cv2
import argparse

from std_msgs.msg import Header 
from cv_bridge import CvBridge
from sensor_msgs.msg import Image






parser = argparse.ArgumentParser()
parser.add_argument("--video-id", type=int, default=4)

args = parser.parse_args()
rospy.loginfo(args)

video_id = args.video_id
cam = cv2.VideoCapture(video_id)


rospy.init_node('camera_streamer')
rate = rospy.Rate(30)
pub = rospy.Publisher('/scrcpy/raw_image', Image, queue_size=1)


bridge = CvBridge()


while not rospy.is_shutdown():
    
    # Capture frame
    ret, frame = cam.read()
    if not ret: continue

    # Convert to ROS image
    ros_image = bridge.cv2_to_imgmsg(frame)

    # Publish
    pub.publish(ros_image)

    rate.sleep()