#!/bin/bash

# Create video loopback device
sudo modprobe v4l2loopback video_nr=3 card_label="scrcpy stream"

# Start streaming to the created video device
scrcpy --video-source=camera --no-audio --camera-id=2 --v4l2-sink=/dev/video3  -m1280 --no-window