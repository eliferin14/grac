import transformations as tf
import numpy as np
from scipy.spatial.transform import Rotation

def normalize_vector(v):
        norm = np.linalg.norm(v)
        return v / norm

class HandLandmarkNormalizer():
    
    def __init__(self):
        pass   
    
    
    
    def __call__(self, landmark_matrix, handedness):
        
        # Extract the point required to calculate the transofrmations:
        # 0 -> wrist base
        # 5 -> index base
        # 17 -> pinky base
        p0 = landmark_matrix[0]
        p5 = landmark_matrix[5]
        p17 = landmark_matrix[17]
        
        # Calculate the hand reference frame
        hand_x = normalize_vector(p5 - p0)
        hand_z = normalize_vector(np.cross( hand_x, (p17-p0) ))
        hand_y = np.cross( hand_z, hand_x )
        assert np.linalg.norm(hand_x) - 1 < 1e-6 and np.linalg.norm(hand_y) - 1 < 1e-6 and np.linalg.norm(hand_z) - 1 < 1e-6
        assert np.dot(hand_x, hand_y) < 1e-6 and np.dot(hand_x, hand_z) < 1e-6 and np.dot(hand_z, hand_y) < 1e-6
        
        # Calculate the relative orientation between world frame and hand frame
        R_hw = np.vstack([hand_x, hand_y, hand_z]) # Each row is the x,y,z vectors of hand frame expressed in world coordinates
        assert R_hw.shape == (3,3)
        assert np.allclose( R_hw.T, np.linalg.inv(R_hw), atol=1e-6 )
        
        # Calculate the translation vector between world frame and hand frame
        t_wh = p0   # Origin of hand frame in world coordinates
        
        # Calculate the cumulative transofrmation
        T_hw = np.eye(4)
        T_hw[:3,:3] = R_hw
        T_hw[:3,3] = -R_hw @ t_wh
        
        # Convert the landmarks tp homogeneous coordinates
        homogeneous_landmarks = np.hstack([landmark_matrix, np.ones((21,1))])
        
        # Apply the transformation
        normalized_homogeneous_landmarks = (T_hw @ homogeneous_landmarks.T).T
        
        # Return to cartesian coordinates
        normalized_landmarks = normalized_homogeneous_landmarks[:,:-1] 
        
        # Assert the normalization is correct
        assert np.allclose(normalized_landmarks[0], np.zeros_like(normalized_landmarks[0]), atol=1e-6)  # Assert the palm base id in the origin of hand frame
        assert np.dot(normalized_landmarks[5]-normalized_landmarks[0], np.array([0,0,1])) < 1e-6        # Assert the palm is in xy plane of hand frame
        assert np.dot(normalized_landmarks[17]-normalized_landmarks[0], np.array([0,0,1])) < 1e-6       # Assert the palm is in xy plane of hand frame
        
        # Scale to obtain a standardized size
        scaling_factor = np.linalg.norm((p5-p0))
        normalized_landmarks /= scaling_factor
        
        # Normalize so that the hand gesture recognition always work with the right hand
        assert handedness == "Right" or handedness == "Left"
        if handedness == 'Left':
            normalized_landmarks[:,2] = -normalized_landmarks[:,2]
        
        return normalized_landmarks
    