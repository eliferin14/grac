import time

class GestureFilter():
    
    def __init__(self,
                 transition_timer = 0.1
                 ):
        
        self.transition_timer = transition_timer
        
        # Initialise old gesture to none
        self.old_gesture = None
        self.candidate_gesture = None
        
        
    def gesture_change_request(self, new_gesture):
        
        self.transition_flag = False
                
        if new_gesture is None:
            return False, None
        
        # Check if the new gesture is different from the old one
        # If they are the same, reset the request
        if self.old_gesture == new_gesture:
            self.candidate_gesture = None
            return self.transition_flag, self.old_gesture
        
        # If they are different, check if the request is new
        # If it is, save the time and the new gesture as candidates
        if self.candidate_gesture is None or self.candidate_gesture != new_gesture:
            self.request_time_start = time.time()
            self.candidate_gesture = new_gesture
            
        # If we are here, it means that a state change is occurring
        # Check how much time has passed since the start of the request and compare it with the timer
        delta_t = time.time() - self.request_time_start
        if delta_t > self.transition_timer:
            # Register the transition to a variable and turn a flag to true
            # Only if there is a previous gesture recorded
            if self.old_gesture is not None:
                self.transition_flag = True
                self.transition = self.old_gesture + " -> " + self.candidate_gesture
            
            # Register the new gesture
            self.old_gesture = self.candidate_gesture
        
        return self.transition_flag, self.old_gesture
        
        
    