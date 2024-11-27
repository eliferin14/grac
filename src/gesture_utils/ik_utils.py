from typing import Any
from moveit_commander import MoveGroupCommander

class IK():
    
    group_commander = MoveGroupCommander("manipulator")
    
    def solve_ik(self, pose) -> Any:
        
        # Set the new pose target for the group
        self.group_commander.set_pose_target(pose)
        
        # Plan without executing
        plan = self.group_commander.plan()
        
        # Get the joint configuration from the last point in the planned trajectory
        point = plan[1].joint_trajectory.points[-1].positions
        
        return point