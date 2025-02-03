#!/usr/bin/env python

import rospy
from std_msgs.msg import String
import roslib.packages
import os
import cv2
import numpy as np

from std_msgs.msg import Header 
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

from message_filters import Subscriber, ApproximateTimeSynchronizer

from gesture_control.msg import draw, plot
from gesture_utils.gesture_detector import GestureDetector
from gesture_utils.framework_selector import GestureInterpreter
from gesture_utils.ros_utils import convert_matrix_to_ROSpoints
from gesture_utils.fps_counter import FPS_Counter
from gesture_utils.drawing_utils import draw_on_frame

# Find the model directory absolute path
model_realtive_path = "src/gesture_utils/training/exported_model"
package_path = roslib.packages.get_pkg_dir('gesture_control')
model_absolute_path = file_path = os.path.join(package_path, model_realtive_path)




# Create arm object
# Don't forget to launch the robot simulator!
from sami.arm import Arm, EzPose
arm = Arm('ur10e_moveit', group='manipulator')










class SubscriberNode():
    
    def __init__(self):
        
        rospy.init_node('single_processing_node')
        
        # Define the detector object
        self.detector = GestureDetector(
            model_absolute_path,
            0.2
        )
        
        # Create the framework selector
        self.interpreter = GestureInterpreter()
        
        # Initialise FPS counter
        self.fps_counter = FPS_Counter()

        # Flag to indicate if processing is ongoing
        self.processing = False 

        # OpenCV bridge
        self.bridge = CvBridge()
        
        # Initialise the subscribers
        self.rgb_sub = Subscriber('/camera/color/image_raw', Image, queue_size=1)
        self.depth_sub = Subscriber('/camera/depth/image_raw', Image, queue_size=1)

        # Initialize the publishers
        self.draw_publisher = rospy.Publisher('draw_topic', draw, queue_size=1)
        
        # Initialise the time synchronization mechanism
        self.ats = ApproximateTimeSynchronizer([self.rgb_sub, self.depth_sub], queue_size=1, slop=0.1)
        self.ats.registerCallback(self.gesture_detection_callback)
        
        
        
    def gesture_detection_callback(self, rgb_msg, depth_msg):
        
        if self.processing:
            rospy.loginfo("Skipping because already busy")
            return
            
        self.processing = True
        
        fps = self.fps_counter.get_fps()
        
        # Convert RGB image to opencv frame
        frame = self.bridge.imgmsg_to_cv2(rgb_msg, 'bgr8')
        frame = cv2.flip(frame, 1)
        
        # Convert depth image to opencv matrix
        depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='32FC1')
        depth_image = cv2.flip(depth_image, 1)
        depth_image_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
        depth_image_uint8 = np.uint8(depth_image_normalized)
        colored_depth_image = cv2.applyColorMap(depth_image_uint8, cv2.COLORMAP_JET)
        
        # Detect 
        self.detector.process(frame)
        
        # Extract gestures
        rh_gesture, lh_gesture = self.detector.get_hand_gestures()
        
        # Call the gesture interpreter
        callback = self.interpreter.interpret_gestures(
            frame=frame,
            fps=fps,
            arm=arm,
            rhg=rh_gesture,
            lhg=lh_gesture,
            rhl=self.detector.right_hand_landmarks_matrix,
            lhl=self.detector.left_hand_landmarks_matrix,
            pl=self.detector.pose_landmarks_matrix
        )
        
        # Execute the selected callback        
        #result = callback()
        
        # Draw stuff on the frame 
        """ draw_on_frame(
            frame=frame,
            rhg=rh_gesture,
            lhg=lh_gesture,
            rhl=self.detector.right_hand_landmarks_matrix,
            lhl=self.detector.left_hand_landmarks_matrix,
            pl=self.detector.pose_landmarks_matrix,
            fps=fps,
            framework_names=self.interpreter.framework_names,
            candidate_framework=self.interpreter.candidate_framework_index,
            selected_framework=self.interpreter.selected_framework_index,
            min_theta=self.interpreter.menu_manager.min_theta,
            max_theta=self.interpreter.menu_manager.max_theta
        )  """       
        
        cv2.imshow("Live feed", frame)
        cv2.imshow("Depth", colored_depth_image)
        cv2.waitKey(1)
            
        self.processing = False
        
        
        
        
if __name__ == "__main__":
    
    sub = SubscriberNode()
    
    rospy.spin()