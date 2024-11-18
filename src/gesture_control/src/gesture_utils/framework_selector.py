import rospy
import numpy as np
from functools import partial

from gesture_utils.frameworks.base_framework import BaseFrameworkManager
from gesture_utils.frameworks.joint_control import JointFrameworkManager
from gesture_utils.frameworks.menu_framework import MenuFrameworkManager

from sami.arm import Arm, EzPose



def dummy_callback():
    return



class FrameworkSelector():
    
    menu_manager = MenuFrameworkManager()
    
    framework_managers = [
        BaseFrameworkManager(),
        BaseFrameworkManager(),
        BaseFrameworkManager(),
        BaseFrameworkManager(),
        #JointFrameworkManager()
    ]
    
    
    
    
    
    def __init__(self):
        
        # The default framework is the base framework
        # NOTE for testing purposes the default is the joint control
        self.selected_framework_manager = self.framework_managers[0]
        self.selected_framework_index, self.candidate_framework_index = 0, 0
        
        for i, f in enumerate(self.framework_managers):
            if f.framework_name == "Base":
                f.framework_name = f"Framework {i}"
                       
        self.framework_names = [ fw.framework_name for fw in self.framework_managers]        
                
        
        
        
        
        
    def interpret_gestures(self, *args, **kwargs):
        
        """This function asks to the selected framework manager for a function to execute. Then returns such function to the caller

        Args:
            rh_gesture (_type_): _description_
            lh_gesture (_type_): _description_

        Returns:
            _type_: _description_
        """        ''''''
        
        #print(kwargs)
        
        # If the left hand is doing the L gesture, call the menu selection
        if kwargs['lhg'] == 'L':
            
            if self.selected_framework_manager != self.menu_manager:
                rospy.loginfo("Open menu")
            
            # Add the framework names list to kwargs
            kwargs['fwn'] = self.framework_names
            
            # Call the interpretation function of the menu
            self.candidate_framework_index, rh_confirmation, callback = self.menu_manager.interpret_gestures(*args, **kwargs)
            
            # If the RH confirmed, change the selected framework
            if rh_confirmation:
                self.selected_framework_index = self.candidate_framework_index
                
                # Select the desired framework
                self.selected_framework_manager = self.framework_managers[self.selected_framework_index]
                
            rospy.loginfo(f"Candidate framework index: {self.candidate_framework_index}, Selected framework index: {self.selected_framework_index}")
            
            
            #rospy.loginfo(f"Selected framework: [{self.selected_framework_index}] -> \'{self.selected_framework_manager.framework_name}\'")
            
            # Return the dummy callback
            return partial(callback)
        
        
        # Call the framework manager and do something
        callback = self.selected_framework_manager.interpret_gestures(*args, **kwargs)
        
        return callback
        