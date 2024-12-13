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
    
    
def draw_centered_text(image, text, center, font_scale, color, thickness=1, font=cv2.FONT_HERSHEY_SIMPLEX):
    """
    Draws text centered at the specified point on an image.

    Parameters:
    - image: The image on which to draw (numpy array).
    - text: The text to draw (string).
    - center: A tuple (x, y) representing the center point for the text (int, int).
    - font_scale: Font scale factor that affects the size of the text (float).
    - color: Text color (BGR tuple, e.g., (255, 255, 255)).
    - thickness: Thickness of the text stroke (int, default is 1).
    - font: Font type (default is cv2.FONT_HERSHEY_SIMPLEX).
    """
    # Get the text size and baseline
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    # Calculate the bottom-left corner of the text based on the center point
    origin_x = center[0] - text_width // 2
    origin_y = center[1] + text_height // 2

    # Draw the text on the image
    cv2.putText(image, text, (origin_x, origin_y), font, font_scale, color, thickness)



def draw_text_with_background(image, text, text_color, bg_color, font_scale, center, thickness=2, font=cv2.FONT_HERSHEY_SIMPLEX):
    """
    Draws text with a background rectangle centered at a specified point on an image.

    Parameters:
    - image: The image on which to draw (numpy array).
    - text: The text to draw (string).
    - text_color: Text color (BGR tuple, e.g., (255, 255, 255)).
    - bg_color: Background rectangle color (BGR tuple, e.g., (0, 0, 0)).
    - font_scale: Font scale factor that affects the size of the text (float).
    - center: A tuple (x, y) representing the center point for the text (int, int).
    - thickness: Thickness of the text stroke (int, default is 1).
    - font: Font type (default is cv2.FONT_HERSHEY_SIMPLEX).
    """
    # Get the text size and baseline
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    # Calculate the bottom-left corner of the text
    origin_x = center[0] - text_width // 2
    origin_y = center[1] + text_height // 2

    # Define the rectangle coordinates
    top_left = (origin_x - 5, origin_y - text_height - baseline - 5)  # Add some padding
    bottom_right = (origin_x + text_width + 5, origin_y + baseline + 5)

    # Draw the background rectangle
    cv2.rectangle(image, top_left, bottom_right, bg_color, -1)  # Filled rectangle
    cv2.rectangle(image, top_left, bottom_right, text_color, thickness)  # Filled rectangle

    # Draw the centered text
    draw_centered_text(image, text, center, font_scale, text_color, thickness, font)




def draw_menu_linear(
    frame,
    names,
    candidate,
    selected,
    hl_pixel,
    dot_size,
    dot_distance,
    candidate_color,
    selected_color,
    
):
    return
    
    
    
def draw_menu_arc(
    frame, 
    framework_names, 
    candidate_framework,
    selected_framework, 
    lhl_pixel, 
    min_theta=np.pi/4, 
    max_theta=np.pi*3/4, 
    scaling=1.3, 
    radius=15, 
    candidate_color=(0,0,255),
    selected_color=(255,128,0)
):
    
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
        
        # Select colors
        fill_color = (255,255,255)
        border_color = (0,0,0)
        
        if (fw_number-i-1) == candidate_framework:
            fill_color = candidate_color
            border_color = (0,0,0)
        
        if (fw_number-i-1) == selected_framework:
            fill_color = selected_color
            border_color = (0,0,0)
        
        cv2.circle(frame, (x,y), radius, fill_color, thickness=-1)
        cv2.circle(frame, (x,y), radius, border_color, thickness=3)
        
    # Draw the name of the selected framework
    draw_text_with_background(frame, framework_names[candidate_framework], text_color=(0,0,0), bg_color=candidate_color, font_scale=0.5, center=(palm_origin[0],palm_origin[1]))
        
        
    
    
    
    







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
    candidate_framework=0,
    min_theta=np.pi/4,
    max_theta=np.pi*3/4
):
    # De-normalize landmarks
    height, width = frame.shape[0], frame.shape[1]
    rhl_pixel = denormalize_landmarks(rhl, width, height)
    lhl_pixel = denormalize_landmarks(lhl, width, height)
    pl_pixel = denormalize_landmarks(pl, width, height)
    
    # Draw pose
    #draw_pose(frame, pl_pixel, point_color=(0,255,0), line_color=(128,128,128))
    
    # Draw hands
    draw_hand(frame, rhl_pixel, point_color=(255,0,0), line_color=(255,255,255))
    draw_hand(frame, lhl_pixel, point_color=(0,0,255), line_color=(255,255,255))
    
    # Add text
    left_padding = 15
    line_thickness = 30
    fontscale = 0.7
    cv2.putText(frame, f"FPS: {fps:.1f}", (left_padding,line_thickness), cv2.FONT_HERSHEY_SIMPLEX, fontScale=fontscale, color=(0,255,0), thickness=2)
    cv2.putText(frame, f"Left: {lhg}", (left_padding,line_thickness*2), cv2.FONT_HERSHEY_SIMPLEX, fontScale=fontscale, color=(0,0,255), thickness=2)
    cv2.putText(frame, f"Right: {rhg}", (left_padding,line_thickness*3), cv2.FONT_HERSHEY_SIMPLEX, fontScale=fontscale, color=(255,0,0), thickness=2)
    cv2.putText(frame, f"Framework: {framework_names[selected_framework]}", (left_padding,line_thickness*4), cv2.FONT_HERSHEY_SIMPLEX, fontScale=fontscale, color=(255,128,0), thickness=2)
    
    # Draw the menu
    #if lhg == 'L': draw_menu_arc(frame, framework_names, candidate_framework, selected_framework, lhl_pixel, min_theta, max_theta)
    
    