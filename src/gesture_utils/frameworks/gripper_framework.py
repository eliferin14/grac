#!/usr/bin/env python3

'''
The gripper framework have to control the opening and the closing of the gripper

The gripper is controlled with a JointGroupPositionController defined in the .yaml file

For the simulation it is in iris_ur10e/ur_e_gazebo/controller/arm_controller.yaml with the name gripper_controller
=> a gripper_controller/command topic is generated
    it accepts as command the target position of the two fingers. How is it defined? Who knows
    maybe I can extend the JointActionFramework
'''


import rospy
import numpy as np
from functools import partial
from gesture_utils.frameworks.base_framework import BaseFrameworkManager

from std_msgs.msg import Float64MultiArray  # Import the message type









class GripperFrameworkmanager(BaseFrameworkManager):
    
    # I assume that if I am here then lhg == 'pick'
    rhg_list = ['pick', 'open']
    
    # Define fingers names
    #joint_names = ['left_finger_joint', 'right_finger_joint']
    
    # Define the topic name
    #topic_name = '/placeholder' if rospy.get_param('/detection_node/live') else '/gripper_controller/command'
    topic_name = '/gripper_controller/command'
    
    pick_msg, release_msg = Float64MultiArray(), Float64MultiArray()
    pick_msg.data = [0,0]
    release_msg.data = [1,1]
    
    
    
    
    def __init__(self):
        
        # Initialise the publisher
        self.publisher = rospy.Publisher(self.topic_name, Float64MultiArray, queue_size=1)    
        
    
    def _publish_msg(self, msg):
        self.publisher.publish(msg)
    
    def pick(self):
        self._publish_msg(self.pick_msg)
    
    def release(self):
        self._publish_msg(self.release_msg)
    
    






if __name__ == "__main__":
    rospy.init_node("gripper_control_node")
    
    gc = GripperFrameworkmanager()
    
    closed = True
    while not rospy.is_shutdown():
        closed = not closed
        if closed:
            gc.release()
        
        rospy.sleep(3)
    
    rospy.spin()