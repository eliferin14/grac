#!/usr/bin/env python

import rospy
from std_msgs.msg import String
import roslib.packages
import os
import cv2

from std_msgs.msg import Header 
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

from gesture_utils.gesture_detector import GestureDetector
from gesture_utils.ros_utils import convert_2Dmatrix_to_ROSpoints, convert_3Dmatrix_to_ROSpoints
from gesture_control.msg import draw

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

def talker():
    
    # Start the node
    rospy.init_node('gesture_detector', anonymous=True)
    rate = rospy.Rate(10)  # 10hz
    
    # Initialize the publisher
    pub = rospy.Publisher('draw_topic', draw, queue_size=10)
    
    # Initialize the bridge
    bridge = CvBridge()
    
    while not rospy.is_shutdown():
        
        # Capture frame
        ret, frame = cam.read()
        if not ret: continue
        frame = cv2.flip(frame, 1)
        
        # Process frame
        detector.process(frame)
        rhl = convert_2Dmatrix_to_ROSpoints(detector.right_hand_landmarks_matrix)
        rhwl = convert_3Dmatrix_to_ROSpoints(detector.right_hand_world_landmarks_matrix)
        lhl = convert_2Dmatrix_to_ROSpoints(detector.left_hand_landmarks_matrix)
        lhwl = convert_3Dmatrix_to_ROSpoints(detector.left_hand_world_landmarks_matrix)
        #print(lhl)
        
        # Convert frame to ROS image
        ros_image = bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        
        # Publish on the draw topic
        draw_msg = draw()
        draw_msg.header = Header()
        draw_msg.frame = ros_image
        draw_msg.rh_2D_landmarks = rhl
        draw_msg.lh_2D_landmarks = lhl
        pub.publish(draw_msg)
        rospy.loginfo("Published an image to /draw_topic")
        
        rate.sleep()
        
    cam.release()






if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass
    

    
    


