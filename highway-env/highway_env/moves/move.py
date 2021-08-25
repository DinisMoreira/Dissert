from highway_env.vehicle.kinematics import Vehicle
from highway_env.vehicle.objects import Landmark

class Move():

    def __init__(self, vehicle: Vehicle, finalGoal: Landmark) -> None:
        self.phase = 1
        self.vehicle = vehicle
        self.firstPhaseGoal = None
        self.secondPhaseGoal = None
        self.finalGoal = finalGoal
        self.path = []
        self.path_index = 0

    def is_vehicle_west_of_final_goal(self) -> bool:
        return self.vehicle.position[0] > self.finalGoal.position[0]

    def is_vehicle_south_of_final_goal(self) -> bool:
        return self.vehicle.position[1] < self.finalGoal.position[1]