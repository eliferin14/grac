import rospy
from functools import partial
import numpy as np

from gesture_utils.frameworks.base_framework import BaseFrameworkManager





def draw_menu(frame, **kwargs):
    print(kwargs['index'])
    return
    





class MenuFrameworkManager(BaseFrameworkManager):
    
    framework_name = "Menu"
    
    min_theta = np.pi * 3/8
    max_theta = np.pi * 5/8
    range_theta = max_theta - min_theta
    
    def interpret_gestures(self, *args, **kwargs):
        
        #print(kwargs['fwn'])
        
        # Extract parameters from kwargs
        rhg = kwargs['rhg']
        lhg = kwargs['lhg']
        rhl = kwargs['rhl']
        lhl = kwargs['lhl']
        fwn = kwargs['fwn']
        
        # Given the number of selectable frameworks, calculate the angle sectors
        fw_number = len(fwn)
        sector_width = self.range_theta / fw_number
        sector_limits = [ i*sector_width for i in range(fw_number) ]
        
        # Calculate angle of the finger
        wrist_base = lhl[0]
        index_tip = lhl[8]
        dx = index_tip[0] - wrist_base[0]
        dy = index_tip[1] - wrist_base[1]
        theta = -1*np.arctan2(dy, dx)
        
        # Choose the sector based on the calculated angle and the limits
        dtheta = theta - self.min_theta
        #rospy.loginfo(f"LH angle: {theta:.2f}, LH delta angle: {dtheta:.2f}")
        s = 0
        while dtheta > (s)*sector_width:
            s += 1
        if s==0: s=1    # Edge case when the angle is outside the range (dtheta < 0)
            
        # Wait for confirmation from right hand
        rh_confirmation = False
        if rhg == 'pick':
            rh_confirmation = True
        
        # "Reverse" the index: 0 is on the left
        index = fw_number - s
        
        # Prepare parameter list for the drawing function
        draw_kwargs = {
            'index':index
        }
        
        return index, rh_confirmation,  partial(
                                            draw_menu,
                                            frame=None,
                                            **draw_kwargs
                                        )