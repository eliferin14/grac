import time

class GestureTransitionManager():
    
    def __init__(self,
                 transition_timer = 0.1
                 ):
        
        self.transition_timer = transition_timer
        
        # Initialise old gesture to none
        self.old_gesture = None
        self.candidate_gesture = None
        
        
    def gesture_change_request(self, new_gesture):
        
        if new_gesture is None:
            return
        
        # Check if the new gesture is different from the old one
        # If they are the same, reset the request
        if self.old_gesture == new_gesture:
            self.candidate_gesture = None
            return self.old_gesture
        
        # If they are different, check if the request is new
        # If it is, save the time and the new gesture as candidates
        if self.candidate_gesture is None or self.candidate_gesture != new_gesture:
            self.request_time_start = time.time()
            self.candidate_gesture = new_gesture
            
        # If we are here, it means that a state change is occurring
        # Check how much time has passed since the start of the request and compare it with the timer
        delta_t = time.time() - self.request_time_start
        if delta_t > self.transition_timer:
            self.old_gesture = self.candidate_gesture
        
        return self.old_gesture
        
        
    