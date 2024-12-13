import numpy as np
from gesture_utils.smoothing.smoothing_base_class import SmoothingAlgorithm

class CurvatureCorrectedMovingAverage(SmoothingAlgorithm):
    def smooth(self, trajectory: np.ndarray) -> np.ndarray:
        """
        Apply CCMA smoothing to the trajectory.
        :param trajectory: Numpy array of shape (n, 3).
        :return: Smoothed trajectory as a numpy array.
        """
        window_size = self.params.get("window_size", 5)
        curvature_threshold = self.params.get("curvature_threshold", 0.1)

        n_points = len(trajectory)
        smoothed_trajectory = np.copy(trajectory)

        # Compute curvature
        d1 = np.gradient(trajectory, axis=0)
        d2 = np.gradient(d1, axis=0)
        norms_d1 = np.linalg.norm(d1, axis=1)
        norms_d2 = np.linalg.norm(d2, axis=1)
        curvature = np.divide(norms_d2, norms_d1**3, out=np.zeros_like(norms_d1), where=norms_d1 != 0)

        # Apply curvature-adaptive smoothing
        for i in range(n_points):
            # Adjust window size based on curvature
            adaptive_window = max(3, int(window_size * (1 if curvature[i] < curvature_threshold else 0.5)))
            half_window = adaptive_window // 2

            # Compute moving average
            start = max(0, i - half_window)
            end = min(n_points, i + half_window + 1)
            smoothed_trajectory[i] = trajectory[start:end].mean(axis=0)

        return smoothed_trajectory