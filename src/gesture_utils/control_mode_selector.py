import rospy
import numpy as np
from functools import partial

from gesture_utils.control_modes.control_mode_interface import ControlModeInterface
from gesture_utils.control_modes.joint_control_mode import JointControlMode
from gesture_utils.control_modes.cartesian_control_mode import CartesianControlMode
from gesture_utils.control_modes.hand_mimic_control_mode import HandMimicControlMode
from gesture_utils.control_modes.control_mode_selector import MenuFrameworkManager
from gesture_utils.visual_menu import MenuHandler
from gesture_utils.control_modes.gripper_control_mode import GripperControlMode



def dummy_callback():
    return



class GestureInterpreter():
    
    main_menu_handler = MenuHandler()
    
    #menu_manager = MenuFrameworkManager()
    
    framework_managers = [
        JointControlMode(),
        CartesianControlMode(),
        CartesianControlMode(use_ee_frame=True),
        HandMimicControlMode()
    ]

    gripper_controller = GripperControlMode()
    
    
    
    
    
    def __init__(self):
        
        # The default framework is the base framework
        # NOTE for testing purposes the default is the joint control
        self.selected_framework_index, self.candidate_framework_index = 3,3
        self.selected_framework_manager = self.framework_managers[self.selected_framework_index]
        
        # Just change the name of empty frameworks
        for i, f in enumerate(self.framework_managers):
            if f.framework_name == "Base":
                f.framework_name = f"Framework {i}"
                       
        # Extract the framework names
        self.framework_names = [ fw.framework_name for fw in self.framework_managers]      
                
        
        
        
        
        
    def interpret_gestures(self, *args, **kwargs):
        
        """This function asks to the selected framework manager for a function to execute. Then returns such function to the caller

        Args:
            rh_gesture (_type_): _description_
            lh_gesture (_type_): _description_

        Returns:
            _type_: _description_
        """        ''''''
        
        # If left hand is L open the framework selection menu
        if kwargs['lhg'] == 'L':
            
            # Call the menu_handler            
            self.selected_framework_index = self.main_menu_handler.menu_iteration(kwargs['lhl'], self.framework_names, kwargs['rhg']=='pick')
            self.selected_framework_manager = self.framework_managers[self.selected_framework_index]
            
            return partial(self.main_menu_handler.draw_menu, frame=kwargs['frame'])
        
        # If left hand is 'pick' call the gripper control framework
        elif kwargs['lhg'] == 'pick':
            pass
            return self.gripper_controller.interpret_gestures(*args, **kwargs)
        
        else:
            self.main_menu_handler.reset()
        
        
        # Call the framework manager and do something
        # NOTE: remember to return func, not func() !!!
        callback = self.selected_framework_manager.interpret_gestures(*args, **kwargs)
        #rospy.logwarn(callback.func.__name__)
        
        # Note: no need to use partial
        return callback
        