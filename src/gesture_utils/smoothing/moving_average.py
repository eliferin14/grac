from gesture_utils.smoothing.smoothing_base_class import SmoothingAlgorithm
import numpy as np








class MovingAverage(SmoothingAlgorithm):
    
    def smooth(self, trajectory):
        
        window_size = self.params.get('window_size', 3)
        
        # Smooth the given trajectory
        smoothed = np.array([
            np.convolve(trajectory[:, i], np.ones(window_size) / window_size, mode='same')          # Use convolution, then normalize to get the average
            for i in range(trajectory.shape[1])
        ]).T
        
        return smoothed 
    
    
    
    
    
    
    
    
    

    
    
        
        
        