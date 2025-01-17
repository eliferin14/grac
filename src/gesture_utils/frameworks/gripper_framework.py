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
from gesture_utils.scripts.wrgripper import CR200Plug

from std_msgs.msg import Float64MultiArray  # Import the message type









class GripperFrameworkmanager(BaseFrameworkManager):
    
    # I assume that if I am here then lhg == 'pick'
    rhg_list = ['pick', 'open']
    
    # Define fingers names
    #joint_names = ['left_finger_joint', 'right_finger_joint']
    
    # Define the topic name
    #topic_name = '/placeholder' if rospy.get_param('/detection_node/live') else '/gripper_controller/command'
    live_mode = False
    try:
        live_mode = rospy.get_param('/detection_node/live')
    except KeyError:
        pass

    sim_topic_name = '/gripper_controller/command'
    
    pick_msg, release_msg = Float64MultiArray(), Float64MultiArray()
    pick_msg.data = [0,0]
    release_msg.data = [1,1]
    
    
    
    
    def __init__(self):
        
        # Initialise the publisher
        self.publisher = rospy.Publisher(self.sim_topic_name, Float64MultiArray, queue_size=1)    

        # Create the gripper communication plugin object
        # See SAMI server source code
        opt = {'host': '10.1.0.2', 'port': 44221}
        self.gp = CR200Plug(opt)
        
    
    def _publish_msg(self, msg):
        self.publisher.publish(msg)
    
    def grip(self):
        rospy.loginfo("Gripping")
        if not self.live_mode:
            self._publish_msg(self.pick_msg)
        else:
            self.gp.grip()
    
    def release(self):
        rospy.loginfo("Releasing")
        if not self.live_mode:
            self._publish_msg(self.release_msg)
        else:
            self.gp.release()
            





    def interpret_gestures(self, *args, **kwargs):
        
        # I assume that lhg == 'pick', so no need to check that

        # If rhg is also 'pick', then grip
        if kwargs['rhg'] == 'pick':
            return partial(self.grip)
            
        # If rhg is 'open', then release
        elif kwargs['rhg'] == 'palm':
            return partial(self.release)

        # Otherwise do nothing
        return partial(self.dummy_callback)
    
    






if __name__ == "__main__":
    rospy.init_node("gripper_control_node")
    
    gc = GripperFrameworkmanager()
    rospy.loginfo(type(gc.gp))
    gc.live_mode = True
    
    closed = True
    while not rospy.is_shutdown():

        gripper_status = gc.gp.get_status()
        rospy.loginfo(f"Gripper status: {gripper_status}")

        closed = not closed
        if closed:
            gc.release()
        else:
            gc.grip()
        
        rospy.sleep(5)
    
    rospy.spin()