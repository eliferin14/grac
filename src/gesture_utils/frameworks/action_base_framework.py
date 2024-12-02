import rospy
import numpy as np
from functools import partial

from gesture_utils.frameworks.base_framework import BaseFrameworkManager

import actionlib
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint




