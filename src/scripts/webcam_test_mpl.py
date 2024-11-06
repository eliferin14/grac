import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from queue import Queue
import threading
import time
import argparse

# Set up argument parsing to allow camera ID selection
parser = argparse.ArgumentParser(description="Capture video and display FPS.")
parser.add_argument('-id', '--camera_id', type=int, help="ID of the camera device (e.g., 0 for /dev/video0, 1 for /dev/video1, etc.)")
args = parser.parse_args()

# Queue to store frames
frame_queue = Queue(maxsize=1)  # This will hold one frame at a time (the latest)

# OpenCV video capture (use camera ID from command line argument)
cap = cv2.VideoCapture(args.camera_id)

# Check if the camera opened successfully
if not cap.isOpened():
    print(f"Error: Couldn't open video device {args.camera_id}.")
    exit()

# Create a Matplotlib figure and axis
fig, ax = plt.subplots()

# Initialize the image object for Matplotlib (empty initially)
ret, frame = cap.read()
if ret:
    height, width, _ = frame.shape
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
else:
    print("Error: Couldn't read frame.")
    exit()

im = ax.imshow(np.zeros((height, width, 3), dtype=np.uint8))

# Variables for FPS calculation
capture_fps = 0
animation_fps = 0
last_capture_time = time.time()
last_animation_time = time.time()

# Function to capture and process frames continuously in a separate thread
def capture_frames():
    global cap, frame_queue, last_capture_time, capture_fps
    while True:
        ret, frame = cap.read()  # Capture the frame
        if ret:
            # Calculate capture FPS (frames per second)
            current_time = time.time()
            capture_fps = 1 / (current_time - last_capture_time)  # Calculate FPS
            last_capture_time = current_time  # Update the last capture time

            # Process the frame (e.g., convert to RGB for Matplotlib)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Add FPS text to the frame
            cv2.putText(frame_rgb, f'Capture FPS: {capture_fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

            # Place the frame into the queue (non-blocking)
            if not frame_queue.full():
                frame_queue.put(frame_rgb)

# Function to update the animation with the latest frame from the queue
def update(frame):
    global frame_queue, animation_fps, last_animation_time
    if not frame_queue.empty():
        latest_frame = frame_queue.get()  # Get the latest frame from the queue

        # Calculate animation FPS (frames per second)
        current_time = time.time()
        animation_fps = 1 / (current_time - last_animation_time)  # Calculate FPS
        last_animation_time = current_time  # Update the last animation time

        # Add animation FPS text under the capture FPS text
        cv2.putText(latest_frame, f'Animation FPS: {animation_fps:.2f}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        im.set_data(latest_frame)  # Update the image with the new frame

    return [im]

# Start the frame capture loop in a separate thread
capture_thread = threading.Thread(target=capture_frames, daemon=True)
capture_thread.start()

# Create the animation (will call the update function periodically)
ani = FuncAnimation(fig, update, interval=5, blit=True)

# Show the animation
plt.show()

# Release the video capture when done
cap.release()
