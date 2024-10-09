import cv2
import argparse

from hand_gestures import Mediapipe_GestureRecognizer

# Command line arguments
parser = argparse.ArgumentParser(description="Hello")
parser.add_argument("--camera_id", type=int, default=1, help="ID of camera device. Run v4l2-ctl --list-devices to get more info")
parser.add_argument("--model_path", type=str, default="gesture_recognizer_original.task", help="Path to the model .task file")
parser.add_argument("--model_mode", type=int, default=1, help="Model running mode: 0=image, 1=video, 2=live")
parser.add_argument("-mhdc", "--min_hand_detection_confidence", type=float, default=0.5, help="min_hand_detection_confidence")
parser.add_argument("-mhpc", "--min_hand_presence_confidence", type=float, default=0.5, help="min_hand_presence_confidence")
parser.add_argument("-mhtc", "--min_hand_tracking_confidence", type=float, default=0.5, help="min_hand_tracking_confidence")

class GRAC():
    
    def __init__(self, path, mode=1, mhdc=0.5, mhpc=0.5, mhtc=0.5):
        
        # Create a gesture recognizer model
        self.mpgr = Mediapipe_GestureRecognizer(
            model_path=path,
            mode=mode,
            min_hand_detection_confidence=mhdc,
            min_hand_presence_confidence=mhpc,
            min_tracking_confidence=mhtc
            )
        
        # Create a pose detector model
        
    def detect(self, frame, timestamp):
        
        if frame is None:
            return
        
        # Detect hands and pose in the frame
        self.mpgr.detect_hands(frame, timestamp)
        
        return frame
    
    def draw_results(self, frame):
        
        # Draw hands
        frame = self.mpgr.draw_results(frame)
        
        # Draw pose
        
        return frame

if __name__ == "__main__":    
    
    args = parser.parse_args()
    
    grac = GRAC(
        path=args.model_path,
        mode=args.model_mode,
        mhdc=args.min_hand_detection_confidence,
        mhpc=args.min_hand_presence_confidence,
        mhtc=args.min_hand_tracking_confidence
        )
    
    # Open the camera live feed and process the frames
    cam = cv2.VideoCapture(args.camera_id)
    assert cam.isOpened()
    
    frame_number = 0
    while cam.isOpened():
        # Capture frame
        ret, frame = cam.read()
        if not ret:
            continue
        
        # Detect hands and pose
        grac.detect(frame, frame_number)
        frame_number += 1
        
        # Draw hands and pose
        frame = grac.draw_results(frame)   
        # 3D plot of hands
        grac.mpgr.plot_hands_3d()  
        
        # Flip image horizontally
        frame = cv2.flip(frame, 1)  
        
        frame = grac.mpgr.draw_labels(frame)
        
        # Display frame
        cv2.imshow("Live feed", frame)            
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
        
    # Release the camera
    cam.release()
    cv2.destroyAllWindows()