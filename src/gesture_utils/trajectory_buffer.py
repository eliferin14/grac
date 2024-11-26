from collections import deque
import rospy
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint




'''
The buffer is used to not have stutters in the movement of the robot
The goal is to have always at least N points
'''





class TrajectoryBuffer():
    
    
    def __init__(self, buffer_length) -> None:
        
        # Initialise the circular buffer with the provided max length
        self.points_buffer = deque(maxlen=buffer_length)
        
    
    
    
    def add_point(self, target, dt):
        
        new_point_t = rospy.Duration(dt)
        first_point_t = rospy.Duration(0)
        
        # Get the time from start of the last point, and add the dt 
        if len(self.points_buffer) > 0:
            first_point_t = self.points_buffer[0].time_from_start
            last_point_t = self.points_buffer[-1].time_from_start
            new_point_t += last_point_t
        
        # Create the point and append the point to the trajectory. When the buffer is full it automatically deletes the oldest point
        point = JointTrajectoryPoint()
        point.positions = target
        point.time_from_start = new_point_t
        self.points_buffer.append(point)
        
        #print(self.points_buffer)
        
        # Update the time sequence (the first point has been deleted, and its time_from_start has to be subtracted from all the other points)
        # Otherwise the time_from_start integrates to oblivion
        if len(self.points_buffer) == self.points_buffer.maxlen:
            for p in self.points_buffer:
                p.time_from_start -= first_point_t 
                pass
        
    
    
    def get_points(self):
        
        # Returns the deque as a list, ready to be used in a goal object
        return list(self.points_buffer)
    
    
    def clear(self):
        # Reset the buffer to an empty list
        self.points_buffer.clear()