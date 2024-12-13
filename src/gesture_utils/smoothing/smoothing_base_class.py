from abc import ABC, abstractmethod
import numpy as np









class SmoothingAlgorithm(ABC):
    
    def __init__(self, params: dict = None):
        """
        Initialize the smoothing algorithm with optional parameters.
        :param params: Dictionary containing algorithm-specific parameters.
        """
        self.params = params if params else {}

    @abstractmethod
    def smooth(self, trajectory: np.ndarray) -> np.ndarray:
        """
        Abstract method to smooth a given trajectory.
        :param trajectory: Numpy array of shape (n, d), where n is the number of points and d is the dimension.
        :return: Smoothed trajectory as a numpy array.
        """
        pass

    def set_params(self, **kwargs):
        """
        Set or update parameters for the algorithm.
        :param kwargs: Algorithm-specific parameter key-value pairs.
        """
        self.params.update(kwargs)
    
    def get_params(self) -> dict:
        """
        Get the current parameters of the algorithm.
        :return: Dictionary of parameters.
        """
        return self.params