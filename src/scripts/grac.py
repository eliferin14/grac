import cv2

from hand_gestures import Mediapipe_GestureRecognizer

class GRAC():
    
    def __init__(self, mode=1):
        
        # Create a gesture recognizer model
        self.mpgr = Mediapipe_GestureRecognizer(mode=mode)
        
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
    
    grac = GRAC()
    
    # Open the camera live feed and process the frames
    cam = cv2.VideoCapture(0)
    
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
        
        # Flip image horizontally
        frame = cv2.flip(frame, 1)  
        
        # Display frame
        cv2.imshow("Live feed", frame)            
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
        
    # Release the camera
    cam.release()
    cv2.destroyAllWindows()