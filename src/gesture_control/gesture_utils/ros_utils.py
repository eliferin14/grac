import rospy
import numpy as np
from geometry_msgs.msg import Point32

# Deprecated
def _convert_2Dmatrix_to_ROSpoints(matrix):
    
    if matrix is None: return []
    
    # Check that the matrix contains 2D points
    #assert matrix.shape[1] == 2
    
    # Create new array of points
    points = []
    
    # Copy coordinates
    for p in matrix:
        point = Point32()
        point.x = p[0]
        point.y = p[1]
        
        points.append(point)
        
    # Assert all points have been copied
    #assert len(points) == matrix.shape[0]
    
    return points



def convert_matrix_to_ROSpoints(matrix):
    
    if matrix is None: return []
    
    # Check that the matrix contains 2D points
    assert matrix.shape[1] == 3
    
    # Create new array of points
    points = []
    
    # Copy coordinates
    for p in matrix:
        point = Point32()
        point.x = p[0]
        point.y = p[1]
        point.z = p[2]
        
        points.append(point)
        
    # Assert all points have been copied
    assert len(points) == matrix.shape[0]
    
    return points



def convert_ROSpoints_to_matrix(points):
    
    matrix = np.empty( (0,3), dtype=np.float32)
    
    for p in points:
        point = np.zeros((1,3))
        point[0,0] = p.x
        point[0,1] = p.y
        point[0,2] = p.z
        
        matrix = np.vstack([matrix, point])
        
    return matrix