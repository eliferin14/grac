import cv2
import argparse
from collections import namedtuple
import matplotlib.pyplot as plt
import threading
from queue import Queue, Full, Empty
import numpy as np
from copy import deepcopy
import time


from gesture_detector import GestureDetector, plot3D
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

# Cosmetics
right_color = (255,0,0) # Blue in BGR
left_color = (0,0,255)  # Red in BGR







# Parse arguments
args = parser.parse_args()
print(args)

# Create the detector object
detector = GestureDetector(
    model_directory=args.gesture_recognizer_model_directory,
    transition_timer=args.gesture_transition_timer
)

# Fps counter
fps_counter = FPS_Counter()

# Gesture interpreter
interpreter = GestureInterpreter()

# Named tuple for text to print on image
frame_text = namedtuple('FrameText', ['name', 'value', 'color'])
rhg, lhg = 0, 0
rhw_posList = []






# Define the semaphores for the three threads
detecting_done = threading.Event()
drawing_done = threading.Event()
plotting_done = threading.Event()

# Other events
exit_event = threading.Event()

# Define the queues for data
frame_data = Queue(maxsize=1)

# Define three functions for the three threads
def detect_wrapper():
    print("Starting detecting thread")
    
    # Open the camera live feed and process the frames
    cam = cv2.VideoCapture(args.camera_id)
    assert cam.isOpened()
    
    # Loop until exit signal is sent
    first_iter = True
    while not exit_event.is_set():
        
        # Wait for the other threads to be done
        if not first_iter:
            #drawing_done.wait()
            #plotting_done.wait()
            pass
        else: 
            first_iter = False
        
        # Reset the detecting event
        detecting_done.clear()
        print("Detecting...")
        
        # Capture
        ret, frame = cam.read()
        # Process
        frame = cv2.flip(frame, 1)
        detector.process(frame, use_threading=True)
        # Push data to the queues
        frame_data.put(frame)
    
        # Signal to others that detection is done
        detecting_done.set()
        
    # Release camera
    cam.release()


def draw_wrapper():
    
    print("Starting drawing thread")
    
    # Loop until exit signal is sent
    while not exit_event.is_set():
        
        # Wait for the detection to be completed
        detecting_done.wait()
        
        # Reset the drawing flag
        drawing_done.clear()
        print("Drawing...")
        
        # Extract the frame from the queue
        frame = frame_data.get()
        
        # Draw the frame on the screen
        cv2.imshow("Live stream", frame)
        
        # If 'q' is pressed, signal to everyone to exit
        if cv2.waitKey(5) & 0xFF == ord('q'):
            exit_event.set()
    
        # Signal to others that the drawing is done
        drawing_done.set()
        
        
    # Close all windows
    cv2.destroyAllWindows()



def plot_wrapper():
    
    # Create the plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.show(block=False)
    #plt.show()
    
    # Main loop 
    while not exit_event.is_set():
        
        # Wait for the detection to be completed
        detecting_done.wait()
        
        # Reset the plotting flag
        plotting_done.clear()
        print("Plotting...")
        time.sleep(0.5)
    
        # Signal to others that the plottingis done
        plotting_done.set()
        


def draw_and_plot():
    
    print("Draw and plot starting")
    
    # Create the plot
    fig = plt.figure()
    frame_ax = fig.add_subplot(121)
    plot_ax = fig.add_subplot(122, projection='3d')
    plt.show(block=False)
    
    while not exit_event.is_set():
        
        # Wait for the detection to be completed
        #detecting_done.wait()
        
        # Reset the plotting flag
        plotting_done.clear()
        print("Plotting...")
        
        # Extract the frame from the queue
        frame = frame_data.get()
        
        # Draw on the frame
        
        # Convert to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Draw the frame on the figure
        frame_ax.imshow(frame_rgb)
        
        
        # Draw stuff
        plt.draw()
        plt.pause(0.01)
        
        # Signal to others that the plottingis done
        plotting_done.set()



# Define the threads
detect_thread = threading.Thread(target=detect_wrapper, daemon=True)
draw_thread = threading.Thread(target=draw_wrapper, daemon=True)
# The plotting happens in the main thread



# Start the threads
detect_thread.start()
#draw_thread.start()
draw_and_plot()


# Wait
detect_thread.join()
#draw_thread.join()















""" 
# Loop 
while cam.isOpened():
    
    # Update fps
    fps = fps_counter.get_fps()
    
    # Capture frame
    ret, frame = cam.read()
    if not ret:
        continue        
    
    # Flip image horizontally
    frame = cv2.flip(frame, 1) 
    
    # Detect landmarks
    detector.process(frame, use_threading=True)    
    rhg, lhg = detector.get_hand_gestures()
    
    # Call the gesture interpreter
    interpreter.interpret(detector.right_hand_data, detector.left_hand_data)
    
    # Draw plot
    #plot3D(ax, detector.pose_landmarks, detector.right_hand_data.landmarks, detector.left_hand_data.landmarks)
    
    # Draw hands and pose
    detector.draw_results(frame, args.draw_hands, args.draw_pose)   
    # 3D plot of hands
    #grac.mpgr.plot_hands_3d()  
    
    # Add info as text        
    text_list = []
    text_list.append(frame_text('FPS', fps, (0,255,0)))
    if rhg is not None: text_list.append(frame_text('Right', rhg, right_color))
    if lhg is not None: text_list.append(frame_text('Left', lhg, left_color))
    detector.add_text(frame, text_list, row_height=30)
    
    
    # Display frame
    cv2.imshow("Live feed", frame)            
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break
    
# Release the camera
cam.release()
cv2.destroyAllWindows() """