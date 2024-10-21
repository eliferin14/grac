import cv2
import argparse
import time
from datetime import datetime #https://www.programiz.com/python-programming/datetime/strftime
from pathlib import Path
import mediapipe as mp

# Command line arguments
parser = argparse.ArgumentParser(description="Custom gesture training")
parser.add_argument("--camera_id", type=int, default=0, help="ID of camera device. Run v4l2-ctl --list-devices to get more info")
#parser.add_argument("--model_mode", type=int, default=1, help="Model running mode: 0=image, 1=video, 2=live")
parser.add_argument("-N", type=int, default=10, help="Number of frames to capture")
parser.add_argument("-T", type=float, default=0.5, help="Time interval between frames")
parser.add_argument("--dataset_path", type=str, default="dataset.csv", help="Dataset path")
parser.add_argument("-g", "--gesture_name", type=str, default="none", help="Name of the gesture")
parser.add_argument("--countdown", type=int, default=5, help="Countdown (seconds) before starting to capture")

# Mediapipe things
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

def landmark_list_to_string(landmark_list):
    if landmark_list is None:
        return None
    
    output = ""
    
    for landmark in landmark_list.landmark:
        #print(landmark)
        output += f"{landmark.x:.4f}," + f"{landmark.y:.4f}," + f"{landmark.z:.4f}," 
        
    # Remove the last comma and add \n charachter
    output = output[:-1]
    output += '\n'
    return output
        

if __name__ == "__main__":
    
    args = parser.parse_args()
    
    # Open camera
    cam = cv2.VideoCapture(args.camera_id)
    assert cam.isOpened()
    
    # Open dataset file
    file = open(args.dataset_path, "a")
    
    with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        max_num_hands=1) as hands:
        
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
            
            success, image = cam.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image)

            # Draw the hand annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            coord_string = None
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
                    
                for hand_world_landmarks in results.multi_hand_world_landmarks:
                    # Convert the results to a string of coordinates
                    coord_string = landmark_list_to_string(hand_world_landmarks)

                    # Add the gesture name
                    line = args.gesture_name + "," + coord_string
                    print(line)
                    
                    # Append to file
                    file.write(line)
                
            # Flip the image horizontally for a selfie-view display.
            cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break            
            
            end_time = time.time()
            while end_time - start_time < args.T:
                end_time = time.time()
            
    cam.release()
    file.close()