
from typing import Tuple
import pickle

from highway_env.vehicle.kinematics import Vehicle
from reeds_shepp_curves import reeds_shepp as rs
from reeds_shepp_curves import utils
from PythonRobotics.PathPlanning.RRTStarReedsShepp import rrt_star_reeds_shepp as rrts
import matplotlib.pyplot as plt
import math
from operator import itemgetter
import itertools
from datetime import datetime, timedelta
import random
from openpyxl import Workbook

from gym.envs.registration import register
from gym import GoalEnv
import numpy as np
from numpy.core._multiarray_umath import ndarray

from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.observation import MultiAgentObservation
from highway_env.road.lane import StraightLane, LineType
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.objects import Landmark
from highway_env.vehicle.objects import Obstacle

from highway_env.moves.move import Move



class ParkingEnv(AbstractEnv, GoalEnv):
    """
    A continuous control environment.

    It implements a reach-type task, where the agent observes their position and speed and must
    control their acceleration and steering so as to reach a given goal.

    Credits to Munir Jojo-Verge for the idea and initial implementation.
    """
    REWARD_WEIGHTS: ndarray = np.array([1, 1, 0, 0, 0.02, 0.02, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]) #1, 0.3, 0, 0, 0.02, 0.02 #px, py, vx, vy, sin(a), cos(a)
    SUCCESS_GOAL_REWARD: float = 0.10 #0.12 by default
    STEERING_RANGE: float = np.deg2rad(45)
    firstFrame = True
    previousAction = []
    currentAction = []
    parking_lines_positions = []
    path_scaling = 0.1 #RS TURNING RADIUS ADJUSTMENT
    path_points_gap = 0.1

    parkingPaths = {}
    steering_diff_threshold = 1 #1 by default
    complete_path_length = 0
    last_time_path_index_evolved = datetime.now()
    lowest_recorded_path_complete_coeficient = 1.0000000001
    print_success = True
    previous_path_long_dist = 0
    previous_path_complete_coeficient = 1
    previous_path_tangent_velocity = 0

    num_of_timesteps_in_episode = 0
    sum_distance_to_path = 0

    last_100_episodes_success_indicator = []

    total_cumulative_reward = 0

    episode_counter = 0
    workbook = None
    worksheet = None
    workbook_file_path = "Sheet/file"

    workbook_testing = None
    worksheet_testing = None
    workbook_testing_file_path = "Sheet/"
    #"Sheets/Final/Results/dummy.xlsx"


    def __init__(self) -> None:
        super().__init__()
        self.move = None

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "KinematicsGoal",
                "features": ['x', 'y', 'vx', 'vy', 'cos_h', 'sin_h',\
                    'current_acceleration', 'front_wheels_heading', 'absolute_velocity',\
                    'path_dist', 'cos_path_ang_diff', 'sin_path_ang_diff', 'path_long_dist',\
                    'next_path_dist', 'cos_next_path_ang_diff', 'sin_next_path_ang_diff', 'next_path_long_dist',\
                    'next_next_path_dist', 'cos_next_next_path_ang_diff', 'sin_next_next_path_ang_diff', 'next_next_path_long_dist',\
                    'next_next_next_path_dist', 'cos_next_next_next_path_ang_diff', 'sin_next_next_next_path_ang_diff', 'next_next_next_path_long_dist'],
                "scales": [100, 100, 5, 5, 1, 1, 5, 1, 40, 50, 1, 1, 25, 50, 1, 1, 25, 50, 1, 1, 25, 50, 1, 1, 25],
                "normalize": False
            },
            "action": {
                "type": "ContinuousAction"
            },
            "simulation_frequency": 15,
            "policy_frequency": 5,
            "duration": 300,
            "screen_width": 1920 * 2,
            "screen_height": 1080 * 2,
            "centering_position": [0.5, 0.5],
            "scaling": 10 * 2*2,
            "controlled_vehicles": 1,
            "collision_reward": -0.1,
            "layoutType": 0,
            "gridSizeX": 6,
            "gridSizeY": 2,
            "gridSpotWidth": 4,
            "gridSpotLength": 8,
            "corridorWidth": 9,
            "orientationMode": 7, #1: FinalGoal // 2: Three phase goal // 3: Step by step path // 4: Path with small gaps // 5 - RS with obstacles // 6 - Proximity to path // 7 - Path Tracking
            "trackRear": 1,
            "randomInitialState": 0,
            "initialPosition": [[20, 0],[20, 1], [20, -1], [19, 0], [19, 1], [19, -1], [-20, 0], [-20, 1], [-20, -1], [-19, 0], [-19, 1], [-19, -1]],#[[20, 0],[-20, 0]],#
            "initialHeading": 0,
            "startingPhase": 2,
            "endingPhase": 3,
            "obstacles": 1,
            "otherVehicles": 1,
            "generateNewPaths": 0,
            "pathsFileName": "paths_6x2",
            "randomPath": 0,
            "goalSpotNumber": 7,
            "initialPositionNumber": 6
        })
        return config

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        if not self.firstFrame:
            self.previousAction = self.currentAction
        self.currentAction = action

        obs, reward, terminal, info = super().step(action)
        is_it_success = self._is_success(obs['achieved_goal'], obs['desired_goal'])
        if isinstance(self.observation_type, MultiAgentObservation):
            if self.config["orientationMode"] == 1 or self.config["orientationMode"] == 6 or self.config["orientationMode"] == 7:
                success = tuple((self._is_success(agent_obs['achieved_goal'], agent_obs['desired_goal'])) for agent_obs in obs)
            elif self.config["orientationMode"] == 2:
                success = tuple((self._is_success(agent_obs['achieved_goal'], agent_obs['desired_goal']) and self.move.phase >= self.config["endingPhase"]) for agent_obs in obs)
            elif self.config["orientationMode"] == 3 or self.config["orientationMode"] == 4 or self.config["orientationMode"] == 5:
                success = tuple((self._is_success(agent_obs['achieved_goal'], agent_obs['desired_goal']) and self.move.path_index >= len(self.move.path)) for agent_obs in obs)

        else:
            if self.config["orientationMode"] == 1 or self.config["orientationMode"] == 6 or self.config["orientationMode"] == 7:
                success = is_it_success
            elif self.config["orientationMode"] == 2:
                success = (is_it_success and self.move.phase >= self.config["endingPhase"])
            elif self.config["orientationMode"] == 3 or self.config["orientationMode"] == 4 or self.config["orientationMode"] == 5:
                success = (is_it_success and self.move.path_index >= len(self.move.path))
            
            
        info.update({"is_success": success})

        if self.config["orientationMode"] == 2:
            if self.move.phase == 1 and is_it_success:
                print("PHASE 1 COMPLETE! MOVE TO PHASE 2")
                self._advance_to_phase_two()
            elif self.move.phase == 2 and is_it_success:
                print("PHASE 2 COMPLETE! MOVE TO PHASE 3")
                self._advance_to_final_phase()
            elif self.move.phase == 3 and is_it_success:
                print("PHASE 3 COMPLETE!")
        
        elif self.config["orientationMode"] == 3 or self.config["orientationMode"] == 4 or self.config["orientationMode"] == 5:
            if is_it_success:
                self.move.path_index += 1
                if self.move.path_index < len(self.move.path):
                    self.goal = Landmark(self.road, [self.move.path[self.move.path_index][0], self.move.path[self.move.path_index][1]], heading=self.move.path[self.move.path_index][2]) #CREATE LANDMARK IN GOAL LANE AND SET IS AS THE GOAL
                    self.road.objects.append(self.goal) #ADD OBJECT TO ROAD

        self.num_of_timesteps_in_episode += 1
        self.firstFrame = False
        return obs, reward, terminal, info

    def _reset(self):
        print("RESET")
        self.total_cumulative_reward = 0
        self.workbook_testing = Workbook()
        self.worksheet_testing = self.workbook_testing.active


        if self.episode_counter == 0:
            self.workbook = Workbook()
            self.worksheet = self.workbook.active
        else:
            self.workbook.save(self.workbook_file_path)

        self.episode_counter += 1
        self.print_success = True

        self.update_steering_diff_threshold()
        self.previousAction = []
        self.currentAction = []
        self.firstFrame = True
        self.last_time_path_index_evolved = datetime.now()
        self.lowest_recorded_path_complete_coeficient = 1.0000000001
        self.previous_path_long_dist = 0
        self.previous_path_complete_coeficient = 1
        self.previous_path_tangent_velocity = 0
        self.num_of_timesteps_in_episode = 0
        self.sum_distance_to_path = 0
        self._create_road(self.config["gridSizeX"], self.config["gridSizeY"])
        if self.config["obstacles"] == 1:
            if self.config["layoutType"] == 1:
                self._create_obstacles()
            elif self.config["layoutType"] == 0:
                self._create_obstacles_classic()

        if self.config["randomPath"] == 1:
            goal_spot = self._select_random_goal_spot()
        else:
            goal_spot = self.road.network.lanes_list()[self.config["goalSpotNumber"]]#self.np_random.choice(self.road.network.lanes_list())

        self._create_vehicles(goal_spot)

        if self.config["otherVehicles"] == 1:
            self.create_dummy_parked_vehicles(goal_spot)

        self._create_move_to_final_goal(goal_spot)


        filename = self.config["pathsFileName"]
        if len(self.parkingPaths) == 0 and (self.config["orientationMode"] == 5 or self.config["orientationMode"] == 6 or self.config["orientationMode"] == 7):
            if self.config["generateNewPaths"] == 1:
                for initial_position in self.config["initialPosition"]:
                    self.generate_parking_paths(1000, initial_position, self.vehicle.heading)
                outfile = open(filename,'wb')
                pickle.dump(self.parkingPaths,outfile)
                outfile.close()

            else:
                infile = open(filename,'rb')
                self.parkingPaths = pickle.load(infile)
                infile.close()
        
        
        if self.config["orientationMode"] == 5 or self.config["orientationMode"] == 6 or self.config["orientationMode"] == 7:
            self.get_path(goal_spot)
            self.complete_path_length = len(self.move.path)

        self.create_landmarks_for_path(self.move.path)
        final_goal_position = goal_spot.position(goal_spot.length/2, 0)
        self.goal = Landmark(self.road, final_goal_position, heading=goal_spot.heading) #CREATE LANDMARK IN GOAL LANE AND SET IS AS THE GOAL
        self.road.objects.append(self.goal)

        if self.config["orientationMode"] == 2:
            if self.config["startingPhase"] >= 3:
                self._advance_to_final_phase()
            elif self.config["startingPhase"] >= 2 or self.is_vehicle_close_to_goal_lane(self.vehicle.position, goal_spot):
                self._advance_to_phase_two()
            else:
                self._advance_to_phase_one()
        elif self.config["orientationMode"] == 3:
            self.goal = Landmark(self.road, [self.move.path[0][0], self.move.path[0][1]], heading=self.move.path[0][2]) #CREATE LANDMARK IN GOAL LANE AND SET IS AS THE GOAL
            self.road.objects.append(self.goal) #ADD OBJECT TO ROAD

        elif self.config["orientationMode"] == 5:
            self.goal = Landmark(self.road, [self.move.path[0][0], self.move.path[0][1]], heading=self.move.path[0][2]) #CREATE LANDMARK IN GOAL LANE AND SET IS AS THE GOAL
            self.road.objects.append(self.goal) #ADD OBJECT TO ROAD
        
        self.firstFrame = True

        #####################
        '''for point1, point2 in zip(self.move.path, self.move.path[1:]):
            print(self.calculate_points_distance(point1, point2))'''
        #####################

    def update_steering_diff_threshold(self):
        if self.steering_diff_threshold > 0.05:
            self.steering_diff_threshold = 0.9997 * self.steering_diff_threshold
        #print(self.steering_diff_threshold)


#region ThreePhaseGoalManagement
    def _advance_to_phase_one(self):
        self.move.phase = 1
        self._create_first_phase_goal()
        self._set_new_immediate_goal(self.move.firstPhaseGoal)
        print("PHASE 1!")

    def _advance_to_phase_two(self):
        self.move.phase = 2
        self._create_second_phase_goal()
        self._set_new_immediate_goal(self.move.secondPhaseGoal)
        print("PHASE 2!")

    def _advance_to_final_phase(self):
        self.move.phase = 3
        self._set_new_immediate_goal(self.move.finalGoal)
        print("PHASE 3!")

    def calculate_first_phase_goal_location(self):
        x_offset = self.vehicle.LENGTH*1.5
        y_offset = self.vehicle.LENGTH + self.config["corridorWidth"]/4

        secondPhaseGoalPosition = self.calculate_second_phase_goal_location()
        if self.move.is_vehicle_south_of_final_goal():
            if self.move.is_vehicle_west_of_final_goal():
                return [secondPhaseGoalPosition[0] + x_offset, secondPhaseGoalPosition[1]-y_offset]
            else:
                return [secondPhaseGoalPosition[0] - x_offset, secondPhaseGoalPosition[1]-y_offset]
                
        else:
            if self.move.is_vehicle_west_of_final_goal():
                return [secondPhaseGoalPosition[0] + x_offset, secondPhaseGoalPosition[1] + y_offset]
            else:
                return [secondPhaseGoalPosition[0] - x_offset, secondPhaseGoalPosition[1] + y_offset]

    def calculate_first_phase_goal_heading(self):
        if self.move.is_vehicle_west_of_final_goal():
            return math.pi
        else:
            return 0
            
    def calculate_second_phase_goal_location(self):
        if self.config["layoutType"] == 1:
            return [self.move.finalGoal.position[0], 0] #only supporting north buffer zone for now
        else:
            if self.move.is_vehicle_south_of_final_goal():
                return [self.move.finalGoal.position[0], self.config["corridorWidth"]/2]
            else:
                return [self.move.finalGoal.position[0], -self.config["corridorWidth"]/2]
    
    def _create_first_phase_goal(self) -> None:
        self.move.firstPhaseGoal = Landmark(self.road, self.calculate_first_phase_goal_location(), heading=self.calculate_first_phase_goal_heading())

    def _create_second_phase_goal(self) -> None:
        self.move.secondPhaseGoal = Landmark(self.road, self.calculate_second_phase_goal_location(), heading=np.pi/2) #CREATE LANDMARK IN GOAL LANE
        
    def _set_new_immediate_goal(self, goal: Landmark) -> None:
        self.goal = goal #SET IT AS THE CURRENT GOAL
        self.road.objects.append(self.goal) #ADD OBJECT TO ROAD

    def _create_move_to_final_goal(self, goalLaneSpot: StraightLane) -> None:
        goalLaneSpot.line_types = [LineType.CONTINUOUS,LineType.CONTINUOUS]  #CHANGE GOAL LANE TO CONTINUOUS
        
        final_goal_position = goalLaneSpot.position(goalLaneSpot.length/2, 0)
        
        self.goal = Landmark(self.road, final_goal_position, heading=goalLaneSpot.heading) #CREATE LANDMARK IN GOAL LANE AND SET IS AS THE GOAL
        #self.road.objects.append(self.goal) #ADD OBJECT TO ROAD

        self.move = Move(self.controlled_vehicles[0], self.goal)

        if self.config["orientationMode"] == 1:
            self.move.phase = 3
        elif self.config["orientationMode"] == 3 or self.config["orientationMode"] == 4:
            if self.is_vehicle_close_to_goal_lane(self.vehicle.position, goalLaneSpot):
                self.move.phase = 3
            self.create_path(final_goal_position, math.degrees(goalLaneSpot.heading))

    def get_path(self, goalLaneSpot: StraightLane):
        if self.config["trackRear"] == 1:
            half_vehicle_length = self.vehicle.LENGTH / 2
            vehicle_rear_position = [self.vehicle.position[0] + half_vehicle_length * math.cos(self.vehicle.heading + math.pi), self.vehicle.position[1] + half_vehicle_length * math.sin(self.vehicle.heading + math.pi)]
            vehicle_position = tuple(vehicle_rear_position)
            
            final_position = []
            if goalLaneSpot.position(goalLaneSpot.length/2, 0)[1] > 0:
                final_position = tuple([goalLaneSpot.position(goalLaneSpot.length/2, 0)[0], goalLaneSpot.position(goalLaneSpot.length/2, 0)[1] - half_vehicle_length])
            else:
                final_position = tuple([goalLaneSpot.position(goalLaneSpot.length/2, 0)[0], goalLaneSpot.position(goalLaneSpot.length/2, 0)[1] + half_vehicle_length])

            pair_origin_goal = tuple([vehicle_position, final_position])
            self.move.path = self.parkingPaths[pair_origin_goal]
            self.vehicle.path = self.move.path
        else:
            final_position = tuple(goalLaneSpot.position(goalLaneSpot.length/2, 0))
            vehicle_position = tuple(self.vehicle.position)
            pair_origin_goal = tuple([vehicle_position, final_position])
            self.move.path = self.parkingPaths[pair_origin_goal]
            self.vehicle.path = self.move.path
    
#endregion

#region EnvironmentCreation
    def _create_road(self, gridSizeX: int = 25, gridSizeY: int = 5) -> None:
        """
        Create a road composed of straight adjacent lanes.

        :param spots: number of spots in the parking
        """
        net = RoadNetwork()
        width = self.config["gridSpotWidth"] #Width of each parking space
        length = self.config["gridSpotLength"] #Length of each parking space
        lt = (LineType.STRIPED, LineType.STRIPED)
        x_offset = 0 #Horizontal distance between parking spots
        y_offset = 0 #Vertical distance between parking spots
        line_points_gap = 0.5
        self.parking_lines_positions = []

        if self.config["layoutType"] == 0: #Classic parking layout
            y_offset = self.config["corridorWidth"]/2
            for k in range(gridSizeX):
                x = (k - gridSizeX // 2) * (width + x_offset) - width / 2
                
                net.add_lane(str(k), str(k+1), StraightLane([x, y_offset], [x, y_offset+length], width=width, line_types=lt))
                net.add_lane(str(k+1), str(k+2), StraightLane([x, -y_offset], [x, -y_offset-length], width=width, line_types=lt))
                #ADD LINE POINTS
                y = y_offset
                while y <= y_offset+length:
                    self.parking_lines_positions.append([x - width/2, y])
                    y += line_points_gap

                y = -y_offset
                while y >= -y_offset-length:
                    self.parking_lines_positions.append([x - width/2, y])
                    y -= line_points_gap
                
                if k == gridSizeX-1:
                    y = y_offset
                    while y <= y_offset+length:
                        self.parking_lines_positions.append([x + width/2, y])
                        y += line_points_gap
                    
                    y = -y_offset
                    while y >= -y_offset-length:
                        self.parking_lines_positions.append([x + width/2, y])
                        y -= line_points_gap

                
        elif self.config["layoutType"] == 1: #Novel Parking Layout
            for k in range(gridSizeX):
                x = (k - gridSizeX // 2) * (width + x_offset) - width / 2
                for l in range(gridSizeY):
                    net.add_lane(str(k*gridSizeY + l), str(k*gridSizeY + l + 1), StraightLane([x, l*length], [x, (l+1)*length], width=width, line_types=lt))
                
                #ADD LINE POINTS
                y = 0
                while y <= gridSizeY*length:
                    self.parking_lines_positions.append([x - width/2, y])
                    y += line_points_gap

                if k == gridSizeX-1:
                    y = 0
                    while y <= gridSizeY*length:
                        self.parking_lines_positions.append([x + width/2, y])
                        y += line_points_gap

        self.road = Road(network=net,
                         np_random=self.np_random,
                         record_history=self.config["show_trajectories"])

    def _create_vehicles(self, goal_spot: StraightLane) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
        self.controlled_vehicles = []

        leftLimitSpawnPosition = - int(self.config["gridSizeX"]/2) * self.config["gridSpotWidth"]
        rightLimitSpawnPosition = int(self.config["gridSizeX"]/2 - 1.5) * self.config["gridSpotWidth"]
        topLimitSpawnPosition = -self.config["corridorWidth"]/2 + self.config["gridSpotLength"]/2
        bottomLimitSpawnPoint = self.config["corridorWidth"]/2 - self.config["gridSpotLength"]/2

        if self.config["layoutType"] == 1:
            topLimitSpawnPosition -= 15
            bottomLimitSpawnPoint -= 15

        xRange = rightLimitSpawnPosition - leftLimitSpawnPosition
        yRange = topLimitSpawnPosition - bottomLimitSpawnPoint

        xPosition = leftLimitSpawnPosition + xRange*self.np_random.rand()
        if self.config["startingPhase"] == 1:
            for i in range(1000):
                xPosition = leftLimitSpawnPosition + xRange*self.np_random.rand()
                if not self.is_vehicle_close_to_goal_lane([xPosition, 0], goal_spot):
                    break
        yPosition = bottomLimitSpawnPoint + yRange*self.np_random.rand()

        if self.config["randomInitialState"] == 1:
            vehicle_position = [xPosition, yPosition]
            vehicle_heading = 2*np.pi*self.np_random.rand()
        else:
            if self.config["randomPath"] == 1:
                vehicle_position = random.choice(self.config["initialPosition"])
                vehicle_heading = self.config["initialHeading"]
            else:
                vehicle_position = self.config["initialPosition"][self.config["initialPositionNumber"]]
                vehicle_heading = self.config["initialHeading"]

        for i in range(self.config["controlled_vehicles"]):
            vehicle = self.action_type.vehicle_class(self.road, vehicle_position, vehicle_heading, 0)
            self.road.vehicles.append(vehicle)
            self.controlled_vehicles.append(vehicle)
    
    def create_dummy_parked_vehicles(self, goal_spot: StraightLane) -> None:
        for spot in self.road.network.lanes_list():
            if self.config["layoutType"] == 0 and not goal_spot == spot:
                vehicle = self.action_type.vehicle_class(self.road, spot.position(self.config["gridSpotLength"]/2,0), math.pi/2, 0)
                self.road.vehicles.append(vehicle)
            else:
                spot_position = spot.position(0,0)
                if self.config["layoutType"] == 1 and not goal_spot.position(0,0)[0] == spot_position[0]:
                    vehicle = self.action_type.vehicle_class(self.road, spot.position(self.config["gridSpotLength"]/2,0), math.pi/2, 0)
                    self.road.vehicles.append(vehicle)
    
    def _create_obstacles(self) -> None:
        #Create row of obstacles at the bottom of parking grid
        xGridBottomRowObstacleOffset = -self.config["gridSizeX"]/2 * self.config["gridSpotWidth"]
        if self.config["gridSizeX"]%2 == 0:
            xGridBottomRowObstacleOffset -= self.config["gridSpotWidth"]/2

        yBottomRowObstaclePosition = (self.config["gridSizeY"] + 0.5) * self.config["gridSpotLength"]

        for i in range(self.config["gridSizeX"]):
            obstacle = Obstacle(self.road, [i*self.config["gridSpotWidth"] + xGridBottomRowObstacleOffset, yBottomRowObstaclePosition], 0, 0)
            self.road.objects.append(obstacle)


        #Create row of obstacles at the top of parking grid
        xGridTopRowObstacleOffset = -(self.config["gridSizeX"]/2 + 2) * self.config["gridSpotWidth"]
        if self.config["gridSizeX"]%2 == 0:
            xGridTopRowObstacleOffset -= self.config["gridSpotWidth"]/2

        yTopRowObstaclePosition = - 3 * self.config["gridSpotLength"]

        for i in range(self.config["gridSizeX"] + 4):
            obstacle = Obstacle(self.road, [i*self.config["gridSpotWidth"] + xGridTopRowObstacleOffset, yTopRowObstaclePosition], 0, 0)
            self.road.objects.append(obstacle)

        #Create collumn of obstacles at the left and right side of parking grid
        xLeftColumnObstaclePosition =  - (self.config["gridSizeX"]/2 + 1) * self.config["gridSpotWidth"]
        if self.config["gridSizeX"]%2 == 0:
            xLeftColumnObstaclePosition -= self.config["gridSpotWidth"]/2

        yLeftColumnObstacleOffset = -self.config["gridSpotLength"] * 3/2

        for i in range(self.config["gridSizeY"] + 2):
            obstacle1 = Obstacle(self.road, [xLeftColumnObstaclePosition, i*self.config["gridSpotLength"] + yLeftColumnObstacleOffset], np.pi/2, 0)
            obstacle2 = Obstacle(self.road, [xLeftColumnObstaclePosition, (i+0.5)*self.config["gridSpotLength"] + yLeftColumnObstacleOffset], np.pi/2, 0)
            self.road.objects.append(obstacle1)
            self.road.objects.append(obstacle2)

        
        xRightColumnObstaclePosition = (self.config["gridSizeX"]/2 + 0.1) * self.config["gridSpotWidth"]
        if self.config["gridSizeX"]%2 == 0:
            xRightColumnObstaclePosition -= self.config["gridSpotWidth"]/2

        yRightColumnObstacleOffset = -self.config["gridSpotLength"] * 3/2

        for i in range(self.config["gridSizeY"] + 2):
            obstacle1 = Obstacle(self.road, [xRightColumnObstaclePosition, i*self.config["gridSpotLength"] + yLeftColumnObstacleOffset], -np.pi/2, 0)
            obstacle2 = Obstacle(self.road, [xRightColumnObstaclePosition, (i+0.5)*self.config["gridSpotLength"] + yLeftColumnObstacleOffset], -np.pi/2, 0)
            self.road.objects.append(obstacle1)
            self.road.objects.append(obstacle2)

    def _create_obstacles_classic(self) -> None:
        #Create row of obstacles at the bottom of parking grid
        xGridBottomRowObstacleOffset = -(self.config["gridSizeX"]/2 + 6) * self.config["gridSpotWidth"]
        if self.config["gridSizeX"]%2 == 0:
            xGridBottomRowObstacleOffset -= self.config["gridSpotWidth"]/2

        yBottomRowObstaclePosition = self.config["gridSpotLength"] + self.config["corridorWidth"]/2 + 2

        for i in range(self.config["gridSizeX"] + 12):
            obstacle = Obstacle(self.road, [i*self.config["gridSpotWidth"] + xGridBottomRowObstacleOffset, yBottomRowObstaclePosition], 0, 0)
            self.road.objects.append(obstacle)


        #Create row of obstacles at the top of parking grid
        xGridTopRowObstacleOffset = -(self.config["gridSizeX"]/2 + 6) * self.config["gridSpotWidth"]
        if self.config["gridSizeX"]%2 == 0:
            xGridTopRowObstacleOffset -= self.config["gridSpotWidth"]/2

        yTopRowObstaclePosition = - (self.config["gridSpotLength"] + self.config["corridorWidth"]/2 + 2)

        for i in range(self.config["gridSizeX"] + 12):
            obstacle = Obstacle(self.road, [i*self.config["gridSpotWidth"] + xGridTopRowObstacleOffset, yTopRowObstaclePosition], math.pi, 0)
            self.road.objects.append(obstacle)

        #Create collumn of obstacles at the left and right side of parking grid
        '''xLeftColumnObstaclePosition =  - (self.config["gridSizeX"]/2 + 1) * self.config["gridSpotWidth"]
        if self.config["gridSizeX"]%2 == 0:
            xLeftColumnObstaclePosition -= self.config["gridSpotWidth"]/2

        yLeftColumnObstacleOffset = - (self.config["gridSpotLength"] + self.config["corridorWidth"]/2 - 1.5)

        for i in range(int((self.config["corridorWidth"] + 2*self.config["gridSpotLength"])/6.5)):
            obstacle1 = Obstacle(self.road, [xLeftColumnObstaclePosition, i*self.config["gridSpotLength"] + yLeftColumnObstacleOffset], np.pi/2, 0)
            obstacle2 = Obstacle(self.road, [xLeftColumnObstaclePosition, (i+0.5)*self.config["gridSpotLength"] + yLeftColumnObstacleOffset], np.pi/2, 0)
            self.road.objects.append(obstacle1)
            self.road.objects.append(obstacle2)

        
        xRightColumnObstaclePosition = (self.config["gridSizeX"]/2 + 0.1) * self.config["gridSpotWidth"]
        if self.config["gridSizeX"]%2 == 0:
            xRightColumnObstaclePosition -= self.config["gridSpotWidth"]/2

        yRightColumnObstacleOffset = - (self.config["gridSpotLength"] + self.config["corridorWidth"]/2 - 1.5)

        for i in range(int((self.config["corridorWidth"] + 2*self.config["gridSpotLength"])/6.5)):
            obstacle1 = Obstacle(self.road, [xRightColumnObstaclePosition, i*self.config["gridSpotLength"] + yLeftColumnObstacleOffset], -np.pi/2, 0)
            obstacle2 = Obstacle(self.road, [xRightColumnObstaclePosition, (i+0.5)*self.config["gridSpotLength"] + yLeftColumnObstacleOffset], -np.pi/2, 0)
            self.road.objects.append(obstacle1)
            self.road.objects.append(obstacle2)'''
#endregion    

#region VehicleLaneOverlap
    def get_vehicle_rear_wheels_positions(self):
        half_vehicle_length = self.vehicle.LENGTH / 2
        half_vehicle_width = self.vehicle.WIDTH / 2
        vehicle_wheels_positions = []

        #vehicle_front_position = [self.position[0] + half_vehicle_length * math.cos(self.heading), self.position[1] + half_vehicle_length * math.sin(self.heading)]
        vehicle_rear_position = [self.vehicle.position[0] + half_vehicle_length * math.cos(self.vehicle.heading + math.pi), self.vehicle.position[1] + half_vehicle_length * math.sin(self.vehicle.heading + math.pi)]

        #front_rigth_wheel_position = [vehicle_front_position[0] + half_vehicle_width * math.cos(vehicle.heading + math.pi/2), vehicle_front_position[1] + half_vehicle_width * math.sin(vehicle.heading + math.pi/2)]
        #front_left_wheel_position = [vehicle_front_position[0] - half_vehicle_width * math.cos(vehicle.heading + math.pi/2), vehicle_front_position[1] - half_vehicle_width * math.sin(vehicle.heading + math.pi/2)]
        rear_rigth_wheel_position = [vehicle_rear_position[0] + half_vehicle_width * math.cos(self.vehicle.heading + math.pi/2), vehicle_rear_position[1] + half_vehicle_width * math.sin(self.vehicle.heading + math.pi/2)]
        rear_left_wheel_position = [vehicle_rear_position[0] - half_vehicle_width * math.cos(self.vehicle.heading + math.pi/2), vehicle_rear_position[1] - half_vehicle_width * math.sin(self.vehicle.heading + math.pi/2)]

        #vehicle_wheels_positions.append(front_rigth_wheel_position)
        #vehicle_wheels_positions.append(front_left_wheel_position)
        vehicle_wheels_positions.append(rear_rigth_wheel_position)
        vehicle_wheels_positions.append(rear_left_wheel_position)

        return rear_rigth_wheel_position, rear_left_wheel_position

    def get_vehicle_wheels_positions(self, vehicle):
        half_vehicle_length = vehicle.LENGTH / 2
        half_vehicle_width = vehicle.WIDTH / 2
        vehicle_wheels_positions = []

        vehicle_front_position = [vehicle.position[0] + half_vehicle_length * math.cos(vehicle.heading), vehicle.position[1] + half_vehicle_length * math.sin(vehicle.heading)]
        vehicle_rear_position = [vehicle.position[0] + half_vehicle_length * math.cos(vehicle.heading + math.pi), vehicle.position[1] + half_vehicle_length * math.sin(vehicle.heading + math.pi)]

        front_rigth_wheel_position = [vehicle_front_position[0] + half_vehicle_width * math.cos(vehicle.heading + math.pi/2), vehicle_front_position[1] + half_vehicle_width * math.sin(vehicle.heading + math.pi/2)]
        front_left_wheel_position = [vehicle_front_position[0] - half_vehicle_width * math.cos(vehicle.heading + math.pi/2), vehicle_front_position[1] - half_vehicle_width * math.sin(vehicle.heading + math.pi/2)]
        rear_rigth_wheel_position = [vehicle_rear_position[0] + half_vehicle_width * math.cos(vehicle.heading + math.pi/2), vehicle_rear_position[1] + half_vehicle_width * math.sin(vehicle.heading + math.pi/2)]
        rear_left_wheel_position = [vehicle_rear_position[0] - half_vehicle_width * math.cos(vehicle.heading + math.pi/2), vehicle_rear_position[1] - half_vehicle_width * math.sin(vehicle.heading + math.pi/2)]

        vehicle_wheels_positions.append(front_rigth_wheel_position)
        vehicle_wheels_positions.append(front_left_wheel_position)
        vehicle_wheels_positions.append(rear_rigth_wheel_position)
        vehicle_wheels_positions.append(rear_left_wheel_position)

        return vehicle_wheels_positions

    def get_vehicle_sides_positions(self, vehicle):
        half_vehicle_length = vehicle.LENGTH / 2
        half_vehicle_width = vehicle.WIDTH / 2
        vehicle_sides_positions = []

        vehicle_front_position = [vehicle.position[0] + half_vehicle_length * math.cos(vehicle.heading), vehicle.position[1] + half_vehicle_length * math.sin(vehicle.heading)]
        vehicle_rear_position = [vehicle.position[0] + half_vehicle_length * math.cos(vehicle.heading + math.pi), vehicle.position[1] + half_vehicle_length * math.sin(vehicle.heading + math.pi)]
        vehicle_left_position = [vehicle.position[0] + half_vehicle_width * math.cos(vehicle.heading + math.pi/2), vehicle.position[1] + half_vehicle_width * math.sin(vehicle.heading + math.pi/2)]
        vehicle_right_position = [vehicle.position[0] + half_vehicle_width * math.cos(vehicle.heading - math.pi/2), vehicle.position[1] + half_vehicle_width * math.sin(vehicle.heading - math.pi/2)]

        vehicle_sides_positions.append(vehicle_front_position)
        vehicle_sides_positions.append(vehicle_rear_position)
        vehicle_sides_positions.append(vehicle_left_position)
        vehicle_sides_positions.append(vehicle_right_position)

        return vehicle_sides_positions

    def calculate_car_overlap_with_lines(self, interval: float):
        car_wheels_points = self.get_vehicle_wheels_positions(self.vehicle)
        car_sides_points = self.get_vehicle_sides_positions(self.vehicle)
        overlaping_wheels_points = 0
        overlaping_side_points = 0
        overlaping_center_point = 0

        for line_point in self.parking_lines_positions:
            for car_wheel_point in car_wheels_points:
                if self.calculate_points_distance(car_wheel_point, line_point) < interval:
                    overlaping_wheels_points += 1
            for car_side_point in car_sides_points:
                if self.calculate_points_distance(car_side_point, line_point) < interval:
                    overlaping_side_points += 2
            if self.calculate_points_distance(self.vehicle.position, line_point) < interval:
                overlaping_center_point += 4
        
        return overlaping_wheels_points + overlaping_side_points + overlaping_center_point
#endregion 
    
#region GoalSpotSelection
    def _select_random_goal_spot(self) -> StraightLane:
        return self.np_random.choice(self.road.network.lanes_list()) #SELECT RANDOM GOAL SPOT LANE
        #return self.road.network.lanes_list()[2]
        #DEBUGGING, GOING FOR 1ST AVAILABLE SPOT

    def _select_top_row_goal_spot(self) -> StraightLane:
        topRowNet = RoadNetwork()
        k = 0
        for lane in self.road.network.lanes_list():
            if lane.start[1] == 0.0:
                topRowNet.add_lane(str(k), str(k+1), lane)
            k += 1

        return self.np_random.choice(topRowNet.lanes_list()) #SELECT RANDOM GOAL LANE FROM FIRST ROW
#endregion 
   
#region PathCreation
    def create_landmarks_for_path(self, path):
        if path == None:
            return
        for p in path:
            self.goal = Landmark(self.road, [p[0], p[1]], heading=p[2]) #CREATE LANDMARK IN GOAL LANE AND SET IS AS THE GOAL
            self.road.objects.append(self.goal) #ADD OBJECT TO ROAD

    def create_path(self, final_goal_position, final_goal_heading):
        # a list of "vectors" (x, y, angle in degrees)
        #ROUTE = [(-2,4,180), (2,4,0), (2,-3,90)]#, (-5,-6,240), (-6, -7, 160), (-7,-1,80)]
        initial_position = (self.vehicle.position[0], self.vehicle.position[1], math.degrees(self.vehicle.heading))
        final_position = (final_goal_position[0], final_goal_position[1], final_goal_heading)
        ROUTE = [initial_position, final_position]

        if self.move.phase == 1:
            first_phase_goal_location = self.calculate_first_phase_goal_location()
            first_phase_goal_position = (first_phase_goal_location[0], first_phase_goal_location[1], self.calculate_first_phase_goal_heading())
            ROUTE.insert(1, first_phase_goal_position)
        
        second_phase_goal_location = self.calculate_second_phase_goal_location()
        second_phase_goal_position = (second_phase_goal_location[0], second_phase_goal_location[1], final_goal_heading)
        ROUTE.insert(-1, second_phase_goal_position)

        #print(ROUTE)
        for i, position in enumerate(ROUTE):
            ROUTE[i] = [ROUTE[i][0] * self.path_scaling, ROUTE[i][1] * self.path_scaling, ROUTE[i][2]]

        full_path = []
        total_length = 0

        for i in range(len(ROUTE) - 1):
            path = rs.get_optimal_path(ROUTE[i], ROUTE[i+1])
            full_path += path
            total_length += rs.path_length(path)

        #print("Shortest path length: {}".format(round(total_length, 2)))

        initial_position, initial_heading = self.vehicle.position, self.vehicle.heading

        for e in full_path:
            #print(e) 
            # e.steering (LEFT/RIGHT/STRAIGHT), e.gear (FORWARD/BACKWARD), e.param (distance)
            if self.config["orientationMode"] == 4:
                total_distance = 0
                while total_distance < e.param:
                    
                    if e.steering.value == 1 or e.steering.value == 2:
                        total_distance += self.path_points_gap
                        turning_center_position = self.calculate_turn_center_position(e.steering.value, 1/self.path_scaling, initial_position, initial_heading)
                        final_relative_turning_position, final_heading = self.calculate_relative_turning_position_and_heading(e.steering.value, e.gear.value, total_distance, 1/self.path_scaling, initial_heading)
                        final_position = [turning_center_position[0] + final_relative_turning_position[0], turning_center_position[1] + final_relative_turning_position[1]]
                        
                    else:
                        total_distance += self.path_points_gap
                        final_position, final_heading = self.calculate_subroute_straight_position_heading(initial_position, initial_heading, total_distance/self.path_scaling, e.gear.value)
                        

                    self.move.path.append([final_position[0], final_position[1], final_heading])
            
            initial_position, initial_heading = self.get_end_of_subroute_landmark_position_heading(initial_position, initial_heading, e)

            self.move.path.append([initial_position[0], initial_position[1], initial_heading])
            self.move.path.append([initial_position[0], initial_position[1], initial_heading])
            '''self.goal = Landmark(self.road, initial_position, heading=initial_heading) #CREATE LANDMARK IN GOAL LANE AND SET IS AS THE GOAL
            self.road.objects.append(self.goal) #ADD OBJECT TO ROAD'''
        
        #self.create_landmarks_for_path(self.move.path)

    def get_end_of_subroute_landmark_position_heading(self, initial_position: list, initial_heading, subroute):
        if subroute.steering.value == 1 or subroute.steering.value == 2:#LEFT #RIGHT
            return self.get_end_of_subroute_turn_position_heading(initial_position, initial_heading, subroute)
        elif subroute.steering.value == 3:#STRAIGHT
            return self.get_end_of_subroute_straight_position_heading(initial_position, initial_heading, subroute)
        else:
            print("INVALID SUBROUTE PATH STEERING")
            return [0,0], 0

        
    def get_end_of_subroute_turn_position_heading(self, initial_position: list, initial_heading, subroute):
        turn_direction = subroute.steering.value #1 = RIGHT, 2 = LEFT
        movement_direction = subroute.gear.value #1 = FORWARD, 2 = BACKWARD
        turning_angle = subroute.param
        turning_radius = 1/self.path_scaling

        turning_center_position = self.calculate_turn_center_position(turn_direction, turning_radius, initial_position, initial_heading)

        final_relative_turning_position, final_heading = self.calculate_relative_turning_position_and_heading(turn_direction, movement_direction, turning_angle, turning_radius, initial_heading)
        
        final_position = [turning_center_position[0] + final_relative_turning_position[0], turning_center_position[1] + final_relative_turning_position[1]]

        return final_position, final_heading

    def calculate_relative_turning_position_and_heading(self, turn_direction, movement_direction, turning_angle, turning_radius, initial_heading):
        final_polar_coordinates_angle = initial_heading
        final_heading = initial_heading
        if turn_direction == 2:
            final_polar_coordinates_angle += math.pi/2
            if movement_direction == 1:
                final_polar_coordinates_angle -= turning_angle
                final_heading -= turning_angle
            else:
                final_polar_coordinates_angle += turning_angle
                final_heading += turning_angle
        else:
            final_polar_coordinates_angle -= math.pi/2
            if movement_direction == 1:
                final_polar_coordinates_angle += turning_angle
                final_heading += turning_angle
            else:
                final_polar_coordinates_angle -= turning_angle
                final_heading -= turning_angle
        
        return [turning_radius * math.cos(final_polar_coordinates_angle), turning_radius * math.sin(final_polar_coordinates_angle)], final_heading
    
    def calculate_turn_center_position(self, turn_direction, turning_radius, initial_position, initial_heading):
        if turn_direction == 2:
            return [initial_position[0] + turning_radius * math.cos(initial_heading - math.pi/2), initial_position[1] + turning_radius * math.sin(initial_heading - math.pi/2)]
        else:
            return [initial_position[0] + turning_radius * math.cos(initial_heading + math.pi/2), initial_position[1] + turning_radius * math.sin(initial_heading + math.pi/2)]


    def get_end_of_subroute_straight_position_heading(self, initial_position, initial_heading, subroute):
        subroute_length = subroute.param/self.path_scaling
        
        return self.calculate_subroute_straight_position_heading(initial_position, initial_heading, subroute_length, subroute.gear.value)

    def calculate_subroute_straight_position_heading(self, initial_position, initial_heading, distance, gear):
        if gear == 1:
            final_position = [initial_position[0] + distance * math.cos(initial_heading), initial_position[1] + distance * math.sin(initial_heading)]
        else:
            final_position = [initial_position[0] - distance * math.cos(initial_heading), initial_position[1] - distance * math.sin(initial_heading)]

        return final_position, initial_heading


    def generate_parking_paths(self, max_iter, vehicle_position, vehicle_heading):
        i=0
        print("GENERATING PATHS")
        for spot in self.road.network.lanes_list():
            goal_position = [spot.position(spot.length/2, 0)[0], spot.position(spot.length/2, 0)[1]]
            
            start = []
            goal = []
            if self.config["trackRear"] == 0:
                start = [vehicle_position[0], vehicle_position[1], vehicle_heading]
                goal = [goal_position[0], goal_position[1], spot.heading_at(spot.position(0,0)[0])]
                
            
            elif self.config["trackRear"] == 1:
                half_vehicle_length = self.vehicle.LENGTH / 2
                vehicle_rear_position = [vehicle_position[0] + half_vehicle_length * math.cos(vehicle_heading + math.pi), vehicle_position[1] + half_vehicle_length * math.sin(vehicle_heading + math.pi)]
                final_position = []
                final_heading = spot.heading_at(spot.position(0,0)[0])
                if spot.position(spot.length/2, 0)[1] > 0:
                    final_position = [spot.position(spot.length/2, 0)[0], spot.position(spot.length/2, 0)[1] - half_vehicle_length]
                    #final_heading = final_heading + math.pi
                else:
                    final_position = [spot.position(spot.length/2, 0)[0], spot.position(spot.length/2, 0)[1] + half_vehicle_length]
                    
                    
                    
                start = [vehicle_rear_position[0], vehicle_rear_position[1], vehicle_heading]
                goal = [final_position[0], final_position[1], final_heading]

            path = self.create_path_for_obstacles_rs(max_iter, [start[0], start[1]], start[2], [goal[0], goal[1]], goal[2], spot)
            pair_origin_goal = [tuple([start[0], start[1]]), tuple([goal[0], goal[1]])]
            self.parkingPaths[tuple(pair_origin_goal)] = path
            i += 1
            #print(pair_origin_goal)
            print(str(i) + "/" + str(len(self.road.network.lanes_list())))

        
    def create_path_for_obstacles_rs(self, max_iter, vehicle_position, vehicle_heading, goal_position, goal_heading, goalLaneSpot: StraightLane):
        show_animation = False
        # ====Search Path with RRT====
        all_wheel_positions = self.get_all_parked_vehicles_wheels_positions_from_empty_goal_spot(goalLaneSpot)
        all_obstacle_positions = self.get_all_obstacles_positions()
        
        obstacleList = []
        '''(5, 5, 1),
            (4, 6, 1),
            (4, 8, 1),
            (4, 10, 1),
            (6, 5, 1),
            (7, 5, 1),
            (8, 6, 1),
            (8, 8, 1),
            (8, 10, 1)
        ]  # [x,y,size(radius)]'''

        if self.config["otherVehicles"] == 1:
            for wheel_position in all_wheel_positions:
                wheel_position_tuple = (wheel_position[0], wheel_position[1], 2.99)
                obstacleList.append(wheel_position_tuple)
        else:
            free_parking_spot_obstacle_tuples = self.get_path_finding_obstacles_positions_for_empty_parking_spots(goalLaneSpot)
            for tuple in free_parking_spot_obstacle_tuples:
                obstacleList.append(tuple)
        
        if self.config["obstacles"] == 1:
            for obstacle_position in all_obstacle_positions:
                obstacle_position_tuple = (obstacle_position[0], obstacle_position[1], 8)
                obstacleList.append(obstacle_position_tuple)
        
        # Set Initial parameters
        start = [vehicle_position[0], vehicle_position[1], vehicle_heading]
        goal = [goal_position[0], goal_position[1], goal_heading]

        rrt_star_reeds_shepp = rrts.RRTStarReedsShepp(start, goal, obstacleList, [-45.0, 45.0], max_iter=max_iter)
        path = rrt_star_reeds_shepp.planning(animation=show_animation)

        if path == None:
            self.move.path = [[goal_position[0], goal_position[1], goal_heading]]
            return

        path.reverse()

        corrected_path = self.correct_path_heading(path, goal_position, goal_heading)
        
        reduced_path = self.reduce_vector_length_by_jumping_x_elements(5, corrected_path)

        # Draw final path
        if path and show_animation:  # pragma: no cover
            #print(reduced_path)
            
            self.create_landmarks_for_path(reduced_path)

            rrt_star_reeds_shepp.draw_graph()
            plt.plot([x for (x, y, yaw) in path], [y for (x, y, yaw) in path], '-r')
            plt.grid(True)
            plt.pause(0.001)
            plt.show()
        
        return reduced_path
    
    def get_path_finding_obstacles_positions_for_empty_parking_spots(self, goal_spot: StraightLane):
        obstacleList = []
        for spot in self.road.network.lanes_list():
            if self.config["layoutType"] == 0 and not goal_spot == spot:
                spot_position_tuple = (spot.position(spot.length/2, 0)[0], spot.position(spot.length/2, 0)[1], 3)
                obstacleList.append(spot_position_tuple)
            else:
                spot_position = spot.position(0,0)
                if self.config["layoutType"] == 1 and not goal_spot.position(0,0)[0] == spot_position[0]:
                    spot_position_tuple = (spot.position(spot.length/2, 0)[0], spot.position(spot.length/2, 0)[1], 3)
                    obstacleList.append(spot_position_tuple)

        return obstacleList
    
    def correct_path_heading(self, path, goal_position, goal_heading):
        modified_path = []
        for previows_point, current_point in zip(path, path[1:]):
            rel_heading = self.calculate_path_heading_from_two_points(previows_point, current_point)
            modified_path.append([current_point[0], current_point[1], rel_heading])

        corrected_path = self.correct_path(modified_path, goal_position, goal_heading)

        return corrected_path
    
    def correct_path(self, modified_path, goal_position, goal_heading):
        correct_path = []
        reverse_heading = False
        i = 0
        while i < len(modified_path):
            if i == len(modified_path) - 1:
                correct_path.append([goal_position[0], goal_position[1], goal_heading])
            else:
                next_point_heading = modified_path[i+1][2]
                curr_point_heading = modified_path[i][2]
                if next_point_heading == 0:
                    del modified_path[i + 1]
                    continue

                if reverse_heading:
                    if modified_path[i][2] < 0:
                        correct_path.append([modified_path[i][0], modified_path[i][1], modified_path[i][2] + math.pi])
                    else:
                        correct_path.append([modified_path[i][0], modified_path[i][1], modified_path[i][2] - math.pi])
                else:
                    correct_path.append(modified_path[i])

                heading_diff = abs(curr_point_heading - next_point_heading)
                #print("curr_heading = " + str(curr_point_heading) + " // next_heading = " + str(next_point_heading) + " // heading_diff = " + str(heading_diff))   
                if heading_diff > math.pi/2 and heading_diff < math.pi*3/2:
                    #print("REVERSE")
                    if reverse_heading:
                        reverse_heading = False
                    else:
                        reverse_heading = True
                
            i += 1

        return correct_path
        
    def correct_direction_reverse_headings(self, path):
        for previows_point, current_point in zip(path, path[1:]):
            heading_dif = previows_point[2] - current_point[2]
            if (heading_dif > math.pi/2 and heading_dif < math.pi*3/2) or (heading_dif < -math.pi/2 and heading_dif > math.pi*3/2):
                if current_point[2] > 0:
                    current_point[2] = current_point[2] - math.pi
                else:
                    current_point[2] = current_point[2] + math.pi
        
        return path          
    
    def reverse_path_heading(self, path):
        if path[-1][1] > 0 and path[-1][2] < 0:
            for p in path:
                p[2] += math.pi
                if p[2] > math.pi:
                    p[2] -= 2*math.pi
        elif path[-1][1] < 0 and path[-1][2] > 0:
            for p in path:
                p[2] -= math.pi
                if p[2] < -math.pi:
                    p[2] += 2*math.pi
        
        return path

    def calculate_path_heading_from_two_points(self, previows_point, current_point):
        position_dif = [current_point[0] - previows_point[0], current_point[1] - previows_point[1]]

        position_dif_angle = math.atan2(position_dif[1], position_dif[0])

        #print(str(previows_point[2]) + " - " + str(position_dif_angle))
        return position_dif_angle

#endregion    

#region Utils
    def is_vehicle_close_to_goal_lane(self, vehicle_position, goal_spot: StraightLane) -> bool:
        return abs(vehicle_position[0] - goal_spot.position(goal_spot.length/2, 0)[0]) < 30
    
    def calculate_points_distance(self, point1, point2):
        x_dif = point1[0] - point2[0]
        y_dif = point1[1] - point2[1]

        return math.sqrt(x_dif*x_dif + y_dif*y_dif)
    
    def reduce_vector_length_by_jumping_x_elements(self, x, path):
        reduced_path = []
        i = 0

        if path == None:
            return reduced_path

        while i < len(path):
            reduced_path.append(path[i])
            i += x
        reduced_path.append(path[-1])

        return reduced_path

    def get_all_parked_vehicles_wheels_positions(self):
        all_wheel_positions = []
        for vehicle in self.road.vehicles:
            if vehicle != self.vehicle:
                vehicle_wheel_positions = self.get_vehicle_wheels_positions(vehicle)
                for position in vehicle_wheel_positions:
                    all_wheel_positions.append(position)

        return all_wheel_positions
    
    def get_all_parked_vehicles_wheels_positions_from_empty_goal_spot(self, goal_spot:StraightLane):
        all_wheel_positions = []
        other_spots_positions = self.get_other_parking_spots_positions_other_than_goal_spot(goal_spot)

        for position in other_spots_positions:
            vehicle_wheel_positions = self.get_parking_spot_wheel_positions(position)
            for wheel_position in vehicle_wheel_positions:
                all_wheel_positions.append(wheel_position)
        
        return all_wheel_positions
    
    def get_parking_spot_wheel_positions(self, position):
        spot_wheel_positions = []
        half_vehicle_length = self.vehicle.LENGTH / 2
        half_vehicle_width = self.vehicle.WIDTH / 2
        vehicle_wheels_positions = []

        vehicle_front_position = [position[0], position[1] + half_vehicle_length]
        vehicle_rear_position = [position[0], position[1] - half_vehicle_length]

        front_rigth_wheel_position = [vehicle_front_position[0] + half_vehicle_width, vehicle_front_position[1]]
        front_left_wheel_position = [vehicle_front_position[0] - half_vehicle_width, vehicle_front_position[1]]
        rear_rigth_wheel_position = [vehicle_rear_position[0] + half_vehicle_width, vehicle_rear_position[1]]
        rear_left_wheel_position = [vehicle_rear_position[0] - half_vehicle_width, vehicle_rear_position[1]]

        vehicle_wheels_positions.append(front_rigth_wheel_position)
        vehicle_wheels_positions.append(front_left_wheel_position)
        vehicle_wheels_positions.append(rear_rigth_wheel_position)
        vehicle_wheels_positions.append(rear_left_wheel_position)

        return vehicle_wheels_positions



    def get_other_parking_spots_positions_other_than_goal_spot(self, goal_spot:StraightLane):
        other_vehicle_positions = []
        for spot in self.road.network.lanes_list():
            if self.config["layoutType"] == 0 and not goal_spot == spot:
                other_vehicle_positions.append(spot.position(self.config["gridSpotLength"]/2,0))
            else:
                spot_position = spot.position(0,0)
                if self.config["layoutType"] == 1 and not goal_spot.position(0,0)[0] == spot_position[0]:
                    other_vehicle_positions.append(spot.position(self.config["gridSpotLength"]/2,0))
        
        return other_vehicle_positions

    def get_all_obstacles_positions(self):
        all_obstacles_positions = []
        for obstacle in self.road.objects:
            if obstacle.position[0] != self.goal.position[0] or obstacle.position[1] != self.goal.position[1]:
                all_obstacles_positions.append(obstacle.position)

        return all_obstacles_positions

    def isLeft(self, point1: list, point2: list, point3) -> bool:
        return ((point2[0] - point1[0])*(point3[1] - point1[1]) - (point2[1] - point1[1])*(point3[0] - point1[0])) > 0

    def get_vehicle_velocity(self, vehicle: Vehicle):
        return math.sqrt(vehicle.velocity[0]**2 + vehicle.velocity[1]**2)


#endregion 

    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: dict, p: float = 0.5) -> float:
        """
        Proximity to the goal is rewarded

        We use a weighted p-norm

        :param achieved_goal: the goal that was achieved
        :param desired_goal: the goal that was desired
        :param dict info: any supplementary information
        :param p: the Lp^p norm used in the reward. Use p<1 to have high kurtosis for rewards in [0, 1]
        :return: the corresponding reward
        """
        max_delta_time_to_progress_in_path = timedelta(seconds = 1)
        numPathTrackingPoints = 2
        path_progression_value_coefficient = 1
        '''if self.config["trackRear"] == 1:
            half_vehicle_length = self.vehicle.LENGTH / 2
            vehicle_rear_position = [self.vehicle.position[0] + half_vehicle_length * math.cos(self.vehicle.heading + math.pi), self.vehicle.position[1] + half_vehicle_length * math.sin(self.vehicle.heading + math.pi)]
            achieved_goal[0] = vehicle_rear_position[0] / 100
            achieved_goal[1] = vehicle_rear_position[1] / 100'''

        if self.config["orientationMode"] == 1:
            return -np.power(np.dot(np.abs(achieved_goal - desired_goal), self.REWARD_WEIGHTS), p)

        elif self.config["orientationMode"] == 7:
            #print(achieved_goal)
            pNorm = -np.power(np.dot(np.abs(achieved_goal - desired_goal), self.REWARD_WEIGHTS), 0.5)
            #vehicle_acceleration = abs(self.vehicle.action["acceleration"]/3.5)
            crashed = any(vehicle.crashed for vehicle in self.controlled_vehicles)
            
            previous_path_length = len(self.move.path)
            
            
            dist_to_path, angle_diff, path_long_dist, index_of_next_point_in_path, self.move.path = self.vehicle.get_distance_to_closest_path_points_to_vehicle(numPathTrackingPoints, self.move.path, self.config["trackRear"])
            
            self.sum_distance_to_path += dist_to_path
            
            #print("LAT_DIST = " + str(dist_to_path) + " // LONG_DIST = " + str(path_long_dist) + " // ANGLE_DIFF = " + str(angle_diff))
            pathReward = (1-dist_to_path)
            path_complete_coeficient = (previous_path_length - index_of_next_point_in_path) / self.complete_path_length
            #print( "path_complete_coeficient = "  + str(path_complete_coeficient))
            #print( "PATH LENGTH = " + str(previous_path_length)  + " // path_complete_coeficient = "  + str(path_complete_coeficient))


            #print(self.vehicle.get_vehicle_velocity())
            self.worksheet_testing.append([dist_to_path, angle_diff, self.vehicle.get_vehicle_velocity()])
            self.workbook_testing.save(self.workbook_testing_file_path + "_" + str(self.episode_counter - 1) + ".xlsx")





            #TANGENT VELOCITY
            if len(self.move.path) > index_of_next_point_in_path+1:
                current_point = self.move.path[index_of_next_point_in_path+1]
            else:
                current_point = self.move.path[-1]
            point3 = [current_point[0], current_point[1]]
            #print(point3)

            point1, point2 = self.get_vehicle_rear_wheels_positions()

            #print(self.isLeft(point1, point2, point3))
            tangent_velocity = 0
            if not self.firstFrame:
                '''steeringDiff = abs(self.currentAction[1] - self.previousAction[1])
                accelerationDiff = abs(self.currentAction[0] - self.previousAction[0])
                #print("accelDiff = " + str(accelerationDiff) + "  //  steerDiff = " + str(steeringDiff))
                if steeringDiff > self.steering_diff_threshold or accelerationDiff > self.steering_diff_threshold:
                    #print("STEERING_DIFF OR ACCEL_DIFF")
                    pathReward = 0'''
                
                if self.previous_path_complete_coeficient != path_complete_coeficient:
                    #print("JUMP = " + str(self.previous_path_tangent_velocity))
                    tangent_velocity = self.previous_path_tangent_velocity

                else:
                    #print(path_long_dist - self.previous_path_long_dist)
                    tangent_velocity = path_long_dist - self.previous_path_long_dist
                    self.previous_path_tangent_velocity = tangent_velocity
                
                if self.isLeft(point1, point2, point3):
                    tangent_velocity = -tangent_velocity

                
                #print(tangent_velocity)

            self.previous_path_long_dist = path_long_dist
            self.previous_path_complete_coeficient = path_complete_coeficient














            
            if self.get_vehicle_velocity(self.vehicle) < 1:
                pathReward = pathReward*self.get_vehicle_velocity(self.vehicle)
            
            #if abs(dist_to_path) > 0.5:
            if abs(dist_to_path) > 1 or abs(angle_diff) > math.pi/2 or abs(path_long_dist) > 2.75:# or datetime.now() > self.last_time_path_index_evolved + max_delta_time_to_progress_in_path:
                pathReward = 0
            
            '''print(pathReward)
            if pathReward == 0:
                if abs(dist_to_path) > 1:
                    print("dist_to_path = " + str(abs(dist_to_path)))
                if abs(angle_diff) > math.pi/2:
                    print("angle_diff = "  + str(abs(angle_diff)))
                if abs(path_long_dist) > 2.75:
                    print("path_long_dist = " + str(abs(path_long_dist)))
                if tangent_velocity < 0:
                    print("tangent_velocity = " + str(tangent_velocity))'''

            
            if path_complete_coeficient < self.lowest_recorded_path_complete_coeficient:
                self.lowest_recorded_path_complete_coeficient = path_complete_coeficient
                self.last_time_path_index_evolved = datetime.now()
                #print("REDUCED PATH: LOWER PATh COEF = " + str(self.lowest_recorded_path_complete_coeficient))
            
            '''if len(self.move.path) > 2 and datetime.now() > self.last_time_path_index_evolved + max_delta_time_to_progress_in_path:
                #del self.move.path[0]
                if len(self.move.path) - (index_of_next_point_in_path+1) > 2:
                    #print("DELETE FIRST N POINTS OF PATH: " + str(index_of_next_point_in_path + 1))
                    self.move.path = self.move.path[-(len(self.move.path) - (index_of_next_point_in_path+1)):]
                #print(len(self.move.path))
                self.last_time_path_index_evolved = datetime.now()
                #print("AUTO REDUCED PATH")'''
            
            #elif abs(vehicle_acceleration) < 0.1:
                #pathReward = 0

            #print(self.move.path[index_of_next_point_in_path])
            
                
            
            #print("pNorm: " + str(pNorm) + " // pathReward: " + str(pathReward))
            #result = pNorm + 0.25 * pathReward
            #print("(len(self.move.path) - index_of_next_point_in_path) / self.complete_path_length = " + str(len(self.move.path) - index_of_next_point_in_path) + "/" + str(self.complete_path_length) + " /// " + str((len(self.move.path) - index_of_next_point_in_path) / self.complete_path_length))
            
            #print(path_complete_coeficient)
            result = pathReward# * (path_progression_value_coefficient + (1-path_progression_value_coefficient)*(1 - path_complete_coeficient))#Enought so that the vehicle is encoraged to follow the path, but not so much as it would be beneficial to just skip the path and go straight to the goal
            #print("PathReward = " + str(pathReward) + " * " + str(0.5 + 0.5*(1 - path_complete_coeficient)) + " = " + str(result))
            #print(pathReward)
            
            #print(result)
            if crashed:
                return 0 #- 0.25 * pathReward + self.config["collision_reward"] + self.config["collision_reward"] * self.get_vehicle_velocity(self.vehicle)#result + self.config["collision_reward"] + self.config["collision_reward"] * self.get_vehicle_velocity(self.vehicle)
                #return result - 0.1 * pathReward + self.config["collision_reward"] + self.config["collision_reward"] * self.get_vehicle_velocity(self.vehicle)#result + self.config["collision_reward"] + self.config["collision_reward"] * self.get_vehicle_velocity(self.vehicle)
            elif pNorm > -self.SUCCESS_GOAL_REWARD:
                self.total_cumulative_reward += result
                return 1.5#0 #+ result
            
            #print("Path Reward: " + str(0.1* pathReward) + " // pNorm: " + str(pNorm) + " // result: " + str(result))
            #print(result)
            '''if tangent_velocity > 1:
                tangent_reward = 0.5
            elif tangent_velocity < 0:
                tangent_reward = tangent_velocity*2
            else:
                tangent_reward = tangent_velocity/2
            result += tangent_reward '''
            if tangent_velocity < 0:
                result = -1
            #print("RESULT = " + str(result) + " // TANGENT REWARD = " + str(tangent_reward) + " // PATH REWARD = " + str(pathReward * (path_progression_value_coefficient + (1-path_progression_value_coefficient)*(1 - path_complete_coeficient))))
            #print(result)
            self.total_cumulative_reward += result
            return result

        else:
            crashed = any(vehicle.crashed for vehicle in self.controlled_vehicles)            
            
            pNorm = -np.power(np.dot(np.abs(achieved_goal - desired_goal), self.REWARD_WEIGHTS), p)

            steeringDiffSquared = 0
            if not self.firstFrame:
                steeringDiff = self.currentAction[1] - self.previousAction[1]
                steeringDiffSquared = steeringDiff*steeringDiff

            result = pNorm - 0.1 * self.calculate_car_overlap_with_lines(0.25) - 0.05 * steeringDiffSquared
            if self.config["orientationMode"] == 6:
                dist_to_path, angle_diff, index_of_next_point_in_path, self.move.path = self.vehicle.get_distance_to_closest_path_points_to_vehicle(numPathTrackingPoints, self.move.path, self.config["trackRear"])
                result -= 0.01 * dist_to_path
            
            if abs(self.vehicle.speed) < 0.5:
                result -= -0.1

            if crashed:
                return result + self.config["collision_reward"]
            elif result > -self.SUCCESS_GOAL_REWARD:
                return result + 10
            
            return result

        

    def _reward(self, action: np.ndarray) -> float:
        obs = self.observation_type.observe()
        #print(obs["observation"])
        obs = obs if isinstance(obs, tuple) else (obs,)
        return sum(self.compute_reward(agent_obs['achieved_goal'], agent_obs['desired_goal'], {})
                     for agent_obs in obs)

    def _is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> bool:
        if self.config["orientationMode"] != 7:
            return self.compute_reward(achieved_goal, desired_goal, {}) > -self.SUCCESS_GOAL_REWARD
        elif self.config["orientationMode"] == 7:
            #print(achieved_goal)
            pNorm = -np.power(np.dot(np.abs(achieved_goal - desired_goal), self.REWARD_WEIGHTS), 0.5)
            return pNorm > -self.SUCCESS_GOAL_REWARD

    def _is_terminal(self) -> bool:
        """The episode is over if the ego vehicle crashed or the goal is reached."""
        
        time = self.steps >= self.config["duration"]
        crashed = any(vehicle.crashed for vehicle in self.controlled_vehicles)
        obs = self.observation_type.observe()
        obs = obs if isinstance(obs, tuple) else (obs,)
        if self.config["orientationMode"] == 1 or self.config["orientationMode"] == 6 or self.config["orientationMode"] == 7:
            success = all((self._is_success(agent_obs['achieved_goal'], agent_obs['desired_goal'])) for agent_obs in obs)
            if success and self.print_success:
                print("SUCCESS")
                self.print_success = False
            
        elif self.config["orientationMode"] == 2:
            success = all((self._is_success(agent_obs['achieved_goal'], agent_obs['desired_goal']) and self.move.phase >= self.config["endingPhase"]) for agent_obs in obs) 
        elif self.config["orientationMode"] == 3 or self.config["orientationMode"] == 4 or self.config["orientationMode"] == 5:
            success = all((self._is_success(agent_obs['achieved_goal'], agent_obs['desired_goal']) and self.move.path_index >= len(self.move.path) ) for agent_obs in obs)

        if time or success or crashed:
            if len(self.last_100_episodes_success_indicator) >= 100:
                    del self.last_100_episodes_success_indicator[0]
            if success:
                self.last_100_episodes_success_indicator.append(1)
            else:
                self.last_100_episodes_success_indicator.append(0)
            
            counter_of_successes = 0
            for ep in self.last_100_episodes_success_indicator:
                if ep == 1:
                    counter_of_successes += 1
            
            self.worksheet.append([self.episode_counter - 1, self.sum_distance_to_path/self.num_of_timesteps_in_episode, counter_of_successes/len(self.last_100_episodes_success_indicator), self.total_cumulative_reward])
            return True
        else:
            return False
        #return time or success #or crashed


class ParkingEnvActionRepeat(ParkingEnv):
    def __init__(self):
        super().__init__({"policy_frequency": 1, "duration": 20})


register(
    id='parking-v0',
    entry_point='highway_env.envs:ParkingEnv',
)

register(
    id='parking-ActionRepeat-v0',
    entry_point='highway_env.envs:ParkingEnvActionRepeat'
)
