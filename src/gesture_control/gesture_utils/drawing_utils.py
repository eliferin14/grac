import cv2
import numpy as np

def denormalize_landmarks(matrix, width, height):
    landmarks_pixel = np.empty((0,2), dtype=np.int16)
    
    for lm in matrix:
        pixel = np.zeros((1,2), dtype=np.int16)
        pixel[0,0] = int( lm[0] * width)
        pixel[0,1] = int( lm[1] * height)
        landmarks_pixel = np.vstack([landmarks_pixel, pixel])
    
    return landmarks_pixel

def draw_landmarks(frame, landmarks, point_color, line_color):
    # Draw all the landmarks
    for lm in landmarks:
        #print((lm[0], lm[1]))
        cv2.circle(frame, (lm[0], lm[1]), radius=3, color=point_color, thickness=-1)
        cv2.circle(frame, (lm[0], lm[1]), radius=4, color=line_color, thickness=1)
        

hand_connections = [
    (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
    (0, 5),  # Thumb to wrist
    (5, 6), (6, 7), (7, 8),  # Index finger
    (5, 9),  # Index finger to wrist
    (9, 10), (10, 11), (11, 12),  # Middle finger
    (9, 13),  # Middle finger to wrist
    (13, 14), (14, 15), (15, 16),  # Ring finger
    (13, 17),  # Ring finger to wrist
    (17, 18), (18, 19), (19, 20),  # Pinky
    (0, 17)
]
def draw_hand(frame, landmarks, point_color, line_color):
    
    # Draw connections
    for (p1, p2) in hand_connections:
        cv2.line(frame, landmarks[p1], landmarks[p2], line_color, 2)
    
    # Draw all the landmarks
    draw_landmarks(frame, landmarks, point_color, line_color)

def draw_pose(frame, landmarks, color):
    return