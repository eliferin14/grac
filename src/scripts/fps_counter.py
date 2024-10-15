import time

class FPS_Counter():
    
    def __init__(self):
        self.t = time.time()
        
    def get_fps(self):
        t_old = self.t
        self.t = time.time()
        
        delta_t = self.t - t_old
        
        self.fps = 1.0 / delta_t
        
        return self.fps