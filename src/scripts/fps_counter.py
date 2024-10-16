import time

class FPS_Counter():
    
    def __init__(self, decay=0.5):
        self.t = time.time()
        self.fps = -1
        self.decay = decay
        
    def get_fps(self):
        t_old = self.t
        self.t = time.time()
        
        delta_t = self.t - t_old
        new_fps = 1.0 / delta_t
        
        self.fps = (1-self.decay)*self.fps + self.decay*new_fps if self.fps > 0 else new_fps
        
        return self.fps