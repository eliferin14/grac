import cv2
import argparse
from collections import namedtuple

from gesture_detector import GestureDetector
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










if __name__ == "__main__":    
    
    args = parser.parse_args()
    print(args)
    
    grac = GestureDetector(
        model_directory=args.gesture_recognizer_model_directory,
        transition_timer=args.gesture_transition_timer
    )
    
    # Open the camera live feed and process the frames
    cam = cv2.VideoCapture(args.camera_id)
    assert cam.isOpened()
    
    # Fps counter
    fps_counter = FPS_Counter()
    
    # Gesture managers
    rightGTR = GestureFilter(transition_timer=args.gesture_transition_timer)
    leftGTR = GestureFilter(transition_timer=args.gesture_transition_timer)
    
    # Gesture interpreter
    interpreter = GestureInterpreter()
    
    # Named tuple for text to print on image
    frame_text = namedtuple('FrameText', ['name', 'value', 'color'])
    rhg, lhg = 0, 0
    rhw_posList = []
    
    
    
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
        grac.process(frame, use_threading=True)    
        rhg, lhg = grac.get_hand_gestures()
        
        # Filter gestures
        #rht, filtered_rhg = rightGTR.gesture_change_request(rhg)
        #lht, filtered_lhg = leftGTR.gesture_change_request(lhg)
        
        # Call the gesture interpreter
        interpreter.interpret(grac.right_hand_data, grac.left_hand_data)
        
        # Draw hands and pose
        grac.draw_results(frame, args.draw_hands, args.draw_pose)   
        # 3D plot of hands
        #grac.mpgr.plot_hands_3d()  
        
        # Add info as text        
        text_list = []
        text_list.append(frame_text('FPS', fps, (0,255,0)))
        if rhg is not None: text_list.append(frame_text('Right', rhg, right_color))
        if lhg is not None: text_list.append(frame_text('Left', lhg, left_color))
        grac.add_text(frame, text_list, row_height=30)
        
        # Display frame
        cv2.imshow("Live feed", frame)            
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
        
    # Release the camera
    cam.release()
    cv2.destroyAllWindows()