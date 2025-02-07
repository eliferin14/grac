#!/usr/bin/env python

import rospy
import csv
from gesture_control.msg import timestamps  # Replace with your actual package name
from datetime import datetime
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--filename', type=str, default="processing_times.csv")
args = parser.parse_args()
print(args)

# File to store data
filename = args.filename

# Check if the file exists, if not, create a new one with headers
def create_file_if_not_exists():
    if not os.path.exists(filename):
        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Timestamp"] + ["hands"] + ["Op1", "Op2", "Op3"])  # Adjust headers if needed
        rospy.loginfo("File created: %s", filename)

# Initialize the file (create if not exists)
create_file_if_not_exists()

# Open file for writing (append mode after initial creation)
file = open(filename, 'a', newline='')
writer = csv.writer(file)

def callback(msg):
    """Callback function to handle incoming messages and save data to a file."""
    timestamp = msg.header.stamp.to_sec()
    execution_times = list(msg.execution_times)
    num_hands = [msg.num_hands]

    # Format timestamp
    time_str = datetime.utcfromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S.%f')

    # Save data
    row = [time_str] + num_hands + execution_times
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