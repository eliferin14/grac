#!/usr/bin/env python3

import rospy
import csv
import roslib
from gesture_control.msg import timestamps  # Replace with your actual package name
from datetime import datetime
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--filename', type=str, default="processing_times.csv")
args = parser.parse_args()
print(args)

# Find the model directory absolute path
data_realtive_path = "data/timestamps"
package_path = roslib.packages.get_pkg_dir('gesture_control')
data_absolute_path = os.path.join(package_path, data_realtive_path)
file_absolute_path = os.path.join(data_absolute_path, args.filename)

# Check if the file exists, if not, create a new one with headers
def create_file_if_not_exists():
    if not os.path.exists(file_absolute_path):
        with open(file_absolute_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Timestamp", "hands", "control_mode", "lhg", "rhg"]+ ["capture", "landmarks", "gestures", "interpret", "drawing", "ik", "jacobian"])  # Adjust headers if needed
        rospy.loginfo("File created: %s", file_absolute_path)

# Initialize the file (create if not exists)
create_file_if_not_exists()

# Open file for writing (append mode after initial creation)
file = open(file_absolute_path, 'a', newline='')
writer = csv.writer(file)

def callback(msg):
    """Callback function to handle incoming messages and save data to a file."""
    timestamp = msg.header.stamp.to_sec()
    execution_times = list(msg.execution_times)
    num_hands = [msg.num_hands]
    control_mode = [msg.control_mode]
    lhg = [msg.lhg]
    rhg = [msg.rhg]

    # Format timestamp
    time_str = [datetime.utcfromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S.%f')]

    # Save data
    row = time_str + num_hands + control_mode + lhg + rhg + execution_times
    writer.writerow(row)

    rospy.loginfo("Saved data: %s", row)
    


def shutdown_hook():
    rospy.loginfo("Closing file before shutdown...")
    file.close()
    

def listener():
    """Initializes the ROS listener node."""
    rospy.init_node('processing_time_listener', anonymous=True)
    rospy.Subscriber("/timestamps_topic", timestamps, callback)
    rospy.on_shutdown(shutdown_hook)
    rospy.spin()

if __name__ == '__main__':
    listener()