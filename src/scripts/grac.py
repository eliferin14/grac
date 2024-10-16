import cv2
import argparse
import mediapipe as mp
from collections import namedtuple

from hand_gestures import Mediapipe_GestureRecognizer
from pose_estimation import Mediapipe_PoseLandmarker
from fps_counter import FPS_Counter
from gesture_transition_manager import GestureTransitionManager

# Command line arguments
parser = argparse.ArgumentParser(description="Hello")
parser.add_argument("--camera_id", type=int, default=0, help="ID of camera device. Run v4l2-ctl --list-devices to get more info")
parser.add_argument("--model_mode", type=int, default=1, help="Model running mode: 0=image, 1=video, 2=live")
parser.add_argument("--hand_model_path", type=str, default="gesture_recognizer_original.task", help="Path to the hand model .task file")
parser.add_argument("-mhdc", "--min_hand_detection_confidence", type=float, default=0.3, help="min_hand_detection_confidence")
parser.add_argument("-mhpc", "--min_hand_presence_confidence", type=float, default=0.3, help="min_hand_presence_confidence")
parser.add_argument("-mhtc", "--min_hand_tracking_confidence", type=float, default=0.3, help="min_hand_tracking_confidence")
parser.add_argument("--pose_model_path", type=str, default="pose_landmarker_full.task", help="Path to the pose model .task file")
parser.add_argument("-mpdc", "--min_pose_detection_confidence", type=float, default=0.5, help="min_pose_detection_confidence")
parser.add_argument("-mppc", "--min_pose_presence_confidence", type=float, default=0.5, help="min_pose_presence_confidence")
parser.add_argument("-mptc", "--min_pose_tracking_confidence", type=float, default=0.5, help="min_pose_tracking_confidence")
parser.add_argument("-gtt", "--gesture_transition_timer", type=float, default=0.3, help="Timer required for a new grsture to be registered")





class GRAC():
    """Wrapper class that combines the hand and pose detection models
    """    ''''''
    
    def __init__(self, hand_model_path, pose_model_path, mode=1, mhdc=0.5, mhpc=0.5, mhtc=0.5, mpdc=0.5, mppc=0.5, mptc=0.5):
        
        # Create a gesture recognizer model
        self.mpgr = Mediapipe_GestureRecognizer(
            model_path=hand_model_path,
            mode=mode,
            min_hand_detection_confidence=mhdc,
            min_hand_presence_confidence=mhpc,
            min_tracking_confidence=mhtc
            )
        # Create a pose detector model
        self.mppl = Mediapipe_PoseLandmarker(
            model_path=pose_model_path,
            mode = mode,
            min_pose_detection_confidence=mpdc,
            min_pose_presence_confidence=mppc,
            min_tracking_confidence=mptc
        )
        
        
        
        
    def detect(self, frame, timestamp):
        """Call both the hand detection and pose estimation models

        Args:
            frame (opencv image): frame where the models will look for hands and pose
            timestamp (int): increasing int required by VIDEO and LIVE_STREAM mode
        """        ''''''
        
        if frame is None:
            self.logger.warning("Empty frame received")
            return
        
        # Convert to mediapipe image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        
        # Detect hands
        self.mpgr.detect_hands(mp_image, timestamp)
        
        # Detect pose
        self.mppl.detect_pose(mp_image, timestamp)
        
        
        
        
    
    def draw_results(self, frame):
        
        # Draw hands
        frame = self.mpgr.draw_results(frame, draw_bb=False)
        
        # Draw pose
        frame = self.mppl.draw_pose(frame)
    
    
    
    def add_text(self, frame, text_list, row_height):
        """Adds text to a frame. 
        The text is passed as a list of frame_text (which is a named tuple)

        Args:
            frame (opencv image): frame where to add the text
            text_list (list[frame_text]): Set of text to be added
            row_height (int): height of each row (pixels) 
        """        ''''''
        
        for i, frame_text in enumerate(text_list):
            
            # If the value is a float, truncate to 2 decimals
            value_str = f"{frame_text.value:.2f}" if type(frame_text.value) == float else f"{frame_text.value}"
            
            # If the name is None, do not use the :
            name_str = "" if frame_text.name is None else f"{frame_text.name}: "
            
            # Draw the text on the frame
            frame = cv2.putText(frame, name_str+value_str, org=(25, (i+1)*row_height), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=frame_text.color, thickness=2, lineType=cv2.LINE_AA)
            
        
    
    
    
    
    
    
    
    

if __name__ == "__main__":    
    
    args = parser.parse_args()
    
    grac = GRAC(
        hand_model_path=args.hand_model_path,
        pose_model_path=args.pose_model_path,
        mode=args.model_mode,
        mhdc=args.min_hand_detection_confidence,
        mhpc=args.min_hand_presence_confidence,
        mhtc=args.min_hand_tracking_confidence,
        mpdc=args.min_pose_detection_confidence,
        mppc=args.min_pose_presence_confidence,
        mptc=args.min_pose_tracking_confidence
        )
    
    # Open the camera live feed and process the frames
    cam = cv2.VideoCapture(args.camera_id)
    assert cam.isOpened()
    
    # Fps counter
    fps_counter = FPS_Counter()
    
    # Gesture managers
    rightGTR = GestureTransitionManager(transition_timer=args.gesture_transition_timer)
    leftGTR = GestureTransitionManager(transition_timer=args.gesture_transition_timer)
    
    # Named tuple for text to print on image
    frame_text = namedtuple('FrameText', ['name', 'value', 'color'])
    
    frame_number = 0
    rhg, lhg = 0, 0
    while cam.isOpened():
        # Update fps
        fps = fps_counter.get_fps()
        
        # Capture frame
        ret, frame = cam.read()
        if not ret:
            continue
        
        # Detect hands and pose
        grac.detect(frame, frame_number)
        frame_number += 1
        
        #print(grac.mpgr.get_point_by_index(0, 0))
        
        # Draw hands and pose
        grac.draw_results(frame)   
        # 3D plot of hands
        #grac.mpgr.plot_hands_3d()  
        
        # Flip image horizontally
        frame = cv2.flip(frame, 1) 
        
        # Extract gestures
        rhg, lhg = grac.mpgr.get_hand_gestures_names()
        
        # Compare the current gesture to the old gesture
        filtered_rhg = rightGTR.gesture_change_request(rhg)
        filtered_lhg = leftGTR.gesture_change_request(lhg)
        
        # Add info as text        
        text_list = []
        text_list.append(frame_text('FPS', fps, (0,255,0)))
        if rhg is not None: text_list.append(frame_text(None, filtered_rhg, (255,0,0)))
        if lhg is not None: text_list.append(frame_text(None, filtered_lhg, (0,0,255)))
        grac.add_text(frame, text_list, row_height=30)
        
        # Display frame
        cv2.imshow("Live feed", frame)            
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
        
    # Release the camera
    cam.release()
    cv2.destroyAllWindows()