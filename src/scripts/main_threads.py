import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from queue import Queue
import threading
import time
import argparse
from copy import deepcopy
from collections import namedtuple

from gesture_detector import GestureDetector, get_bounding_cube, pose_connections
from fps_counter import FPS_Counter
from gesture_filter import GestureFilter
from gesture_interpreter import GestureInterpreter




# Command line arguments
parser = argparse.ArgumentParser(description="Hello")
parser.add_argument("-id", "--camera_id", type=int, default=3, help="ID of camera device. Run v4l2-ctl --list-devices to get more info")
parser.add_argument("-grmd", "--gesture_recognizer_model_directory", type=str, default="training/exported_model", help="Path to the gesture recognition model")
parser.add_argument("-gtt", "--gesture_transition_timer", type=float, default=0.5, help="Timer required for a new grsture to be registered")
parser.add_argument("--draw_hands", type=bool, default=True, help="If true draw the hands landmarks on the output frame")
parser.add_argument("--draw_pose", type=bool, default=True, help="If true draw the pose landmarks on the output frame")
args = parser.parse_args()

# Cosmetics
right_color = (255,0,0) # Blue in BGR
left_color = (0,0,255)  # Red in BGR

# Named tuple for text to print on image
frame_text = namedtuple('FrameText', ['name', 'value', 'color'])




# Various objects 
detector = GestureDetector(
        model_directory=args.gesture_recognizer_model_directory,
        transition_timer=args.gesture_transition_timer
    )

rightGTR = GestureFilter(transition_timer=args.gesture_transition_timer)
leftGTR = GestureFilter(transition_timer=args.gesture_transition_timer)

interpreter = GestureInterpreter(detector.labels_path)

capture_fps_counter = FPS_Counter()
animation_fps_counter = FPS_Counter()





# Queue to store frames
data_queue = Queue(maxsize=1)  # This will hold one frame at a time (the latest)

# OpenCV video capture (use camera ID from command line argument)
cam = cv2.VideoCapture(args.camera_id)

# Check if the camera opened successfully
if not cam.isOpened():
    print(f"Error: Couldn't open video device {args.camera_id}.")
    exit()
    
    
    
    

# Create a Matplotlib figure and axis
fig, _ = plt.subplots(figsize=(15,7))
frame_ax = fig.add_subplot(121)
plot_ax = fig.add_subplot(122, projection='3d')
frame_ax.axis('off')

# Initialize the image object for Matplotlib (empty initially)
ret, frame = cam.read()
if ret:
    height, width, _ = frame.shape
    frame_ax.set_xlim(0, width)
    frame_ax.set_ylim(height, 0)
else:
    print("Error: Couldn't read frame.")
    exit()

# Initialize image
im = frame_ax.imshow(np.zeros((height, width, 3), dtype=np.uint8))


# Set title and labels
plot_ax.set_xlabel('x')
plot_ax.set_ylabel('y')
plot_ax.set_zlabel('z')

# Initialize 3d plot stuff
line_collection = Line3DCollection([], colors='gray')
pose_lines = plot_ax.add_collection3d(line_collection)
pose_scatter = plot_ax.scatter([], [], [], c='g', marker='o')
cube_scatter = plot_ax.scatter([], [], [], c='g', marker='o', s=1)







# Function to capture and process frames continuously in a separate thread
def capture_frames():
    global cam, data_queue, detector, rightGTR, leftGTR, interpreter, capture_fps_counter
    
    # The loop is killed automatically when the main thread terminates
    while True:
        
        # Update fps
        capture_fps = capture_fps_counter.get_fps()
        
        # Capture the frame and flip
        ret, frame = cam.read()
        if not ret: continue
        frame = cv2.flip(frame, 1)      
        
        # Detect landmarks and gestures
        detector.process(frame, use_threading=True)      
        rhg, lhg = detector.get_hand_gestures()
        
        # Interpret gestures
        interpreter.interpret(detector.right_hand_data, detector.left_hand_data, detector.pose_landmarks)
        
        # Build the data object
        data = {
            'frame': deepcopy(frame),
            'capture_fps': capture_fps,
            'rhg': rhg,
            'lhg': lhg,
            'pose_lms': detector.pose_landmarks
        }

        # If the queue has not been emptied by the animator, flush it 
        if data_queue.full():
            data_queue.get()
            
        # Update the queue
        data_queue.put(data)
            
            
            
            

# Function to update the animation with the latest frame from the queue
def update(f):
    global im, data_queue, animation_fps_counter, detector, pose_scatter, pose_lines, cube_scatter
    
    if not data_queue.empty():
        
        # Update fps
        animation_fps = animation_fps_counter.get_fps()
        
        # Extract data 
        data = data_queue.get()
        frame = data['frame']
        rhg = data['rhg']
        lhg = data['lhg']
        capture_fps = data['capture_fps']
        
        # Draw stuff
        detector.draw_results(frame, args.draw_hands, args.draw_pose)

        # Add animation FPS text under the capture FPS text
        
        # Add info as text        
        text_list = []
        text_list.append(frame_text('Capture FPS', capture_fps, (0,255,0)))
        text_list.append(frame_text('Animation FPS', animation_fps, (0,255,0)))
        if rhg is not None: text_list.append(frame_text('Right', rhg, right_color))
        if lhg is not None: text_list.append(frame_text('Left', lhg, left_color))
        detector.add_text(frame, text_list, row_height=30)
        #cv2.putText(frame, f'Animation FPS: {animation_fps:.2f}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Convert to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Update the image with the new frame
        im.set_data(frame_rgb)  
        
        
        
        
        # Get landmark coordinates from the detector
        pose_lms = data['pose_lms']
        pose_x = pose_lms[:,0]
        pose_y = pose_lms[:,1]
        pose_z = pose_lms[:,2]
        #print(data['pose_lms'].shape)
        
        # Get cube
        corners = get_bounding_cube(pose_x, pose_y, pose_z, 1)
        
        # Create the collection of lines
        lines = []
        for start, end in pose_connections:
            lines.append([
                [pose_x[start], pose_y[start], pose_z[start]],
                [pose_x[end], pose_y[end], pose_z[end]]
            ])
        line_collection.set_segments(lines)
        
        # Update data        
        #pose_scatter.set_data(pose_x[11:], pose_y[11:])
        #pose_scatter.set_3d_properties(pose_z[11:])
        pose_scatter._offsets3d = (pose_x[11:], pose_y[11:], pose_z[11:])
        cube_scatter._offsets3d = (corners[:, 0], corners[:, 1], corners[:, 2])
        
        plot_ax.set_xlim(corners[0,0], corners[1,0])
        plot_ax.set_ylim(corners[0,1], corners[2,1])
        plot_ax.set_zlim(corners[0,2], corners[4,2])

    return [im, pose_lines, pose_scatter, cube_scatter, line_collection]











# Start the frame capture loop in a separate thread
capture_thread = threading.Thread(target=capture_frames, daemon=True)
capture_thread.start()

# Create the animation (will call the update function periodically)
ani = FuncAnimation(fig, update, interval=100, blit=False)

# Define a key press event to stop the animation
def on_key(event):
    if event.key == 'q':  # Close when 'q' is pressed
        ani.event_source.stop()  # Stop the animation
        plt.close(fig)           # Close the figure window

# Connect the event to the figure
fig.canvas.mpl_connect('key_press_event', on_key)

# Show the animation
plt.show()

# Release the video capture when done
cam.release()
