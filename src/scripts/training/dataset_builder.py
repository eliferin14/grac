# https://ai.google.dev/edge/mediapipe/solutions/customization/gesture_recognizer

import cv2
import argparse
import time
from datetime import datetime #https://www.programiz.com/python-programming/datetime/strftime
from pathlib import Path
import mediapipe as mp

mp_hands = mp.solutions.hands
hand_landmarker = mp_hands.Hands(
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            max_num_hands=1
        )

# Command line arguments
parser = argparse.ArgumentParser(description="Custom gesture training")
parser.add_argument("--camera_id", type=int, default=0, help="ID of camera device. Run v4l2-ctl --list-devices to get more info")
#parser.add_argument("--model_mode", type=int, default=1, help="Model running mode: 0=image, 1=video, 2=live")
parser.add_argument("-N", type=int, default=10, help="Number of frames to capture")
parser.add_argument("-T", type=float, default=0.1, help="Time interval between frames")
parser.add_argument("--dataset_path", type=str, default="dataset", help="Dataset path")
parser.add_argument("-gn", "--gesture_name", type=str, default="none", help="Name of the gesture")
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
    i = 0
    while i < args.N:
        start_time = time.time()
        now = datetime.now()
        
        # Capture frame
        ret, frame = cam.read()
        if not ret:
            print("Skipping frame")
            continue   
        
        # Flip image horizontally
        frame = cv2.flip(frame, 1) 
        
        # Display frame
        cv2.imshow("Live feed", frame)  
        
        # Detect hands
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = None
        results = hand_landmarker.process(rgb_frame)   
        
        if results.multi_hand_landmarks:
            
            for hand_landmarks in results.multi_hand_landmarks:
                # Get all landmark x, y coordinates
                x_coords = [landmark.x for landmark in hand_landmarks.landmark]
                y_coords = [landmark.y for landmark in hand_landmarks.landmark]

                # Get bounding box coordinates
                min_x, max_x = min(x_coords), max(x_coords)
                min_y, max_y = min(y_coords), max(y_coords)

                # Convert normalized coordinates to pixel values
                image_height, image_width, _ = frame.shape
                
                min_x = int(min_x * image_width)
                min_y = int(min_y * image_height)
                max_x = int(max_x * image_width)
                max_y = int(max_y * image_height)
                
                x_center = int( (min_x + max_x) / 2 )
                y_center = int( (min_y + max_y) / 2 )
                width = max( (max_x - min_x), (max_y - min_y) ) * 1.5
                
                y1 = int(y_center - width/2)
                y2 = int(y_center + width/2)
                x1 = int(x_center - width/2)
                x2 = int(x_center + width/2)
                
                if y1<0 or x1<0 or y2>image_height or x2>image_width: break
                
                frame_crop = frame[y1:y2, x1:x2]
        
                # Save to selected location
                img_name = path_string + args.gesture_name + "_" + now.strftime("%Y%m%d_%H%M%S") + "_" + str(i) + ".jpg"
                print(img_name)
                ret = cv2.imwrite(img_name, frame_crop)
                
                cv2.imshow("Cropped hand", frame_crop)   
            
                i += 1
        
        
        # Escape          
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
        
        end_time = time.time()
        while end_time - start_time < args.T:
            end_time = time.time()
    
    cam.release()
    cv2.destroyAllWindows()
    

