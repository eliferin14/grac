#!/usr/bin/env python

import rospy
import numpy as np

from gesture_control.msg import draw
from gesture_utils.ros_utils import convert_ROSpoints_to_matrix
from gesture_utils.framework_selector import FrameworkSelector

#from iris_sami.src.sami.arm import Arm, EzPose
from sami.arm import Arm, EzPose







interpreter = FrameworkSelector()

arm = Arm('ur10e_moveit', group='manipulator')



def interpretation_callback(msg):
    
    # Extract data
    ros_image = msg.frame
    rhl_ros = msg.rh_2D_landmarks
    lhl_ros = msg.lh_2D_landmarks
    pl_ros = msg.pose_2D_landmarks
    rh_gesture = msg.rh_gesture
    lh_gesture = msg.lh_gesture
    
    # Ask for a function to execute
    callback = interpreter.interpret_gestures(arm, rh_gesture, lh_gesture)
    rospy.loginfo(f"Selected function {callback.func.__name__}()")
    
    # Execute the function
    result = callback()
    #print(result)
    
    
    




def listen_detection_topic():
    
    # Initialize node
    rospy.init_node("interpretation")
    
    # Subscribe to the draw topic
    rospy.Subscriber('/draw_topic', draw, interpretation_callback)
    
    # Keep the listener spinning
    rospy.spin()
    
    
    

if __name__ == "__main__":
    try:
        listen_detection_topic()
    except rospy.ROSInterruptException:
        pass