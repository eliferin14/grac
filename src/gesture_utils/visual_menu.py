import cv2
import numpy as np

from gesture_utils.drawing_utils import draw_text_with_background









class MenuHandler():
    
    initialised = False
    
    selected_index = 0
    candidate_index = -1
    
    #
    def init_menu(self, hand_landmarks, item_names, item_distance=0.05):
        
        # Save the names as state variable
        self.names = item_names
        
        # Define the anchor point (index tip)
        self.anchor_point = hand_landmarks[8]
        
        # Calculate the items coordinates (pixel)
        self.n_items = len(item_names)     # Number of items to consider
        items_total_width = (self.n_items-1) * item_distance  
        y = self.anchor_point[1] - 2*item_distance     # All items are on the same horizontal line, a bit above the anchor point
        
        self.item_locations = []
        for i in range(self.n_items):
            x = self.anchor_point[0] - items_total_width/2 + i*item_distance
            self.item_locations.append([x,y])
            
        print(self.item_locations)
        
        return
    
    
    
    
    def reset(self):
        self.initialised = False
    
    
    
    
    def menu_iteration(self, hand_landmarks, item_names, set_selected=False):
        
        # The first time call the init_menu function
        if not self.initialised:
            self.init_menu(hand_landmarks, item_names)
            self.initialised = True
        
        # TODO: remove z coordinate if present
        index_tip = hand_landmarks[8][:2]
        
        # Loop to find the closest neighbour
        min_distance = 100000
        self.candidate_index = -1
        for i, point in enumerate(self.item_locations):
            
            distance = np.linalg.norm(index_tip - point)
            
            if distance < min_distance:
                min_distance = distance
                self.candidate_index = i
                
        assert self.candidate_index >= 0
        assert self.candidate_index < self.n_items
        
        # If told so, change selected item
        if set_selected:
            self.selected_index = self.candidate_index
            
        return self.selected_index
        
        
                   
        
        
    
    
    
    
    def draw_menu(self, frame, radius=10, selected_color=(255,127,0), candidate_color=(0,0,255)):
        
        height, widht = frame.shape[0], frame.shape[1]
        
        # Draw items
        for i, center in enumerate(self.item_locations):
            
            cx = int(center[0]*widht)
            cy = int(center[1]*height)
            
            # Select color
            fill_color = (255,255,255)
            border_color = (0,0,0)
            
            if i == self.candidate_index:
                fill_color = candidate_color
            if i == self.selected_index:
                fill_color = selected_color
            
            # Draw circles
            cv2.circle(frame, (cx, cy), radius, fill_color, -1)
            cv2.circle(frame, (cx, cy), radius, border_color, int(radius/5.0))
            
            # Draw text
            if i == self.candidate_index:
                draw_text_with_background(frame, self.names[i], (0,0,0), fill_color, 0.7, (cx, cy-50))
        
        return 