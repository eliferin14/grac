# https://ai.google.dev/edge/mediapipe/solutions/customization/gesture_recognizer

import cv2
import argparse
import time
from datetime import datetime #https://www.programiz.com/python-programming/datetime/strftime
from pathlib import Path

# Command line arguments
parser = argparse.ArgumentParser(description="Custom gesture training")
parser.add_argument("--camera_id", type=int, default=2, help="ID of camera device. Run v4l2-ctl --list-devices to get more info")
#parser.add_argument("--model_mode", type=int, default=1, help="Model running mode: 0=image, 1=video, 2=live")
parser.add_argument("-N", type=int, default=10, help="Number of frames to capture")
parser.add_argument("-T", type=float, default=0.1, help="Time interval between frames")
parser.add_argument("--dataset_path", type=str, default="dataset", help="Dataset path")
parser.add_argument("-g", "--gesture_name", type=str, default="none", help="Name of the gesture")
parser.add_argument("--countdown", type=int, default=5, help="Countdown (seconds) before starting to capture")

if __name__ == "__main__":
    
    # Parse arguments
    args = parser.parse_args()
    print(args)
    
    # Create the directory
    path_string = args.dataset_path + "/" + args.gesture_name + "/"
    Path(path_string).mkdir(parents=True, exist_ok=True)
    
    # Open camera
    cam = cv2.VideoCapture(args.camera_id)
    assert cam.isOpened()
    
    while True:
        # Capture frame
        ret, frame = cam.read()
        if not ret:
            print("Skipping frame")
            continue 
        
        # Display frame
        cv2.imshow("Live feed", frame)            
        if cv2.waitKey(5) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
    
    # Countdown
    for i in range(args.countdown):
        print(f"Countdown: {args.countdown - i}")
        time.sleep(1)
    
    # Loop to capture N frames
    for i in range(args.N):
        start_time = time.time()
        now = datetime.now()
        
        # Capture frame
        ret, frame = cam.read()
        if not ret:
            print("Skipping frame")
            continue        
        
        # Display frame
        cv2.imshow("Live feed", frame)            
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
        
        # Save to selected location
        img_name = path_string + args.gesture_name + "_" + now.strftime("%Y%m%d_%H%M%S") + "_" + str(i) + ".jpg"
        print(img_name)
        ret = cv2.imwrite(img_name, frame)
        
        end_time = time.time()
        while end_time - start_time < args.T:
            end_time = time.time()
    
    cam.release()
    cv2.destroyAllWindows()
    

