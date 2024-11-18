import cv2
import numpy as np
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands

def denormalize_landmarks(matrix, width, height):
    """Converts landmarks coordinates from normalized image coordinates to pixel coordinates

    Args:
        matrix (_type_): _description_
        width (_type_): _description_
        height (_type_): _description_

    Returns:
        _type_: _description_
    """    ''''''
    landmarks_pixel = np.empty((0,2), dtype=np.int16)
    
    for lm in matrix:
        pixel = np.zeros((1,2), dtype=np.int16)
        pixel[0,0] = int( lm[0] * width)
        pixel[0,1] = int( lm[1] * height)
        landmarks_pixel = np.vstack([landmarks_pixel, pixel])
    
    return landmarks_pixel

def draw_landmarks(frame, landmarks, point_color, line_color, blacklist=[] ):
    """Draw the lanmdarks on the image

    Args:
        frame (opencv image): _description_
        landmarks (np matrix): pixel coordinates of the landmarks
        point_color (_type_): _description_
        line_color (_type_): _description_
    """    ''''''
    # Draw all the landmarks
    for i, lm in enumerate(landmarks):
        
        # Skip unwanted landmarks
        if i in blacklist: continue
        cv2.circle(frame, (lm[0], lm[1]), radius=3, color=point_color, thickness=-1)
        cv2.circle(frame, (lm[0], lm[1]), radius=4, color=line_color, thickness=1)
        
        
        
        
hand_connections = list(mp_hands.HAND_CONNECTIONS)

def draw_hand(frame, landmarks, point_color, line_color):
    """Draw the hand landmarks and connections

    Args:
        frame (_type_): _description_
        landmarks (_type_): _description_
        point_color (_type_): _description_
        line_color (_type_): _description_
    """    ''''''
    
    if landmarks.shape[0] == 0: return 
    
    # Draw connections
    for (p1, p2) in hand_connections:
        cv2.line(frame, landmarks[p1], landmarks[p2], line_color, 2)
    
    # Draw all the landmarks
    draw_landmarks(frame, landmarks, point_color, line_color, [])





pose_connections = pose_connections = list(mp_pose.POSE_CONNECTIONS)
pose_landmarks_blacklist = [0,1,2,3,4,5,6,7,8,9,10,17,18,19,20,21,22]

def draw_pose(frame, landmarks, point_color, line_color):
    """Draw the pose landmarks and connections

    Args:
        frame (_type_): _description_
        landmarks (_type_): _description_
        point_color (_type_): _description_
        line_color (_type_): _description_
    """    
    
    if landmarks.shape[0] == 0: return 
    
    # Draw connections
    for (p1, p2) in pose_connections:
        
        # Skip unwanted landmarks
        if p1 in pose_landmarks_blacklist or p2 in pose_landmarks_blacklist:
            continue
        
        cv2.line(frame, landmarks[p1], landmarks[p2], line_color, 2)
    
    # Draw all the landmarks
    draw_landmarks(frame, landmarks, point_color, line_color, pose_landmarks_blacklist)
    
    
    
    
    
def draw_menu(frame, framework_names, selected_framework, lhl_pixel, min_theta=np.pi/4, max_theta=np.pi*3/4, scaling=1.3, radius=15, color=(0,0,0), selected_color=(0,0,255)):
    
    if len(lhl_pixel) == 0: return
    
    #
    range_theta = max_theta - min_theta
        
    # Given the number of selectable frameworks, calculate the angle sectors
    fw_number = len(framework_names)
    if fw_number == 0: return
    
    sector_width = range_theta / fw_number
    sector_limits = [ i*sector_width for i in range(fw_number) ]
    
    # Calculate the length of the index and the desired distance
    palm_origin = lhl_pixel[0]
    index_tip = lhl_pixel[8]
    l = scaling * np.linalg.norm( index_tip-palm_origin )
    
    # Calculate the coordinates where to draw the circles
    for i, name in enumerate(framework_names):
        
        theta = min_theta + sector_width/2 + i*sector_width
        
        x = palm_origin[0] + int( l * np.cos(theta) )
        y = palm_origin[1] - int( l * np.sin(theta) )   # y axis is positive downwards!
        #print((selected_framework, x,y))
        
        # Draw circles
        fill_color = selected_color if (fw_number-i-1) == selected_framework else color
        border_color = selected_color if (fw_number-i-1) != selected_framework else color
        
        cv2.circle(frame, (x,y), radius, fill_color, thickness=-1)
        cv2.circle(frame, (x,y), radius, border_color, thickness=3)
        
    # Draw the name of the selected framework
    cv2.putText(frame, framework_names[selected_framework], (palm_origin[0],palm_origin[1]), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,0,255), thickness=2)
        
        
    
    
    
    







def draw_on_frame(
    frame,
    rhl=[],
    lhl=[],
    pl=[],
    rhg='',
    lhg='',
    fps='',
    framework_names=[],
    selected_framework=0,
    min_theta=np.pi/4,
    max_theta=np.pi*3/4
):
    # De-normalize landmarks
    height, width = frame.shape[0], frame.shape[1]
    rhl_pixel = denormalize_landmarks(rhl, width, height)
    lhl_pixel = denormalize_landmarks(lhl, width, height)
    pl_pixel = denormalize_landmarks(pl, width, height)
    
    # Draw pose
    draw_pose(frame, pl_pixel, point_color=(0,255,0), line_color=(128,128,128))
    
    # Draw hands
    draw_hand(frame, rhl_pixel, point_color=(255,0,0), line_color=(255,255,255))
    draw_hand(frame, lhl_pixel, point_color=(0,0,255), line_color=(255,255,255))
    
    # Add text
    cv2.putText(frame, f"FPS: {fps:.1f}", (50,50), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,255,0), thickness=2)
    cv2.putText(frame, f"Left: {lhg}", (50,100), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,0,255), thickness=2)
    cv2.putText(frame, f"Right: {rhg}", (50,150), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255,0,0), thickness=2)
    
    # Draw the menu
    draw_menu(frame, framework_names, selected_framework, lhl_pixel, min_theta, max_theta)
    
    