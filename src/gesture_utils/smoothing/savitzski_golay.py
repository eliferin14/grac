from scipy.signal import savgol_filter
import numpy as np

from gesture_utils.smoothing.smoothing_base_class import SmoothingAlgorithm

class SavitzkyGolaySmoothing(SmoothingAlgorithm):
    def smooth(self, trajectory: np.ndarray) -> np.ndarray:
        window_size = self.params.get('window_size', 5)
        poly_order = self.params.get('poly_order', 2)
        
        smoothed = np.array([
            savgol_filter(trajectory[:, i], window_size, poly_order)
            for i in range(trajectory.shape[1])
        ]).T
        return smoothed