import rospy
import numpy as np
import time
import argparse

from gesture_control.control_modes.hand_mimic_control_mode import HandMimicControlMode

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--filename', type=str, default='test.npy')

rospy.init_node('trajectory_simulator')





hmcm = HandMimicControlMode()