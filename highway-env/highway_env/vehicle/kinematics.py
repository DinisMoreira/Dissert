import math
from operator import itemgetter
from typing import Union, Optional
import numpy as np
from collections import deque

from highway_env import utils
from highway_env.road.road import Road, LaneIndex
from highway_env.vehicle.objects import RoadObject, Obstacle, Landmark
from highway_env.types import Vector
from numpy import linalg


class Vehicle(RoadObject):

    """
    A moving vehicle on a road, and its kinematics.

    The vehicle is represented by a dynamical system: a modified bicycle model.
    It's state is propagated depending on its steering and acceleration actions.
    """

    COLLISIONS_ENABLED = True
    """ Enable collision detection between vehicles """

    LENGTH = 5.0
    """ Vehicle length [m] """
    WIDTH = 2.0
    """ Vehicle width [m] """
    DEFAULT_SPEEDS = [23, 25]
    """ Range for random initial speeds [m/s] """
    MAX_SPEED = 40.
    """ Maximum reachable speed [m/s] """

    def __init__(self,
                 road: Road,
                 position: Vector,
                 heading: float = 0,
                 speed: float = 0,
                 path: list = []):
        super().__init__(road, position, heading, speed)
        self.action = {'steering': 0, 'acceleration': 0}
        self.crashed = False
        self.impact = None
        self.log = []
        self.history = deque(maxlen=30)
        self.current_acceleration = 0
        self.front_wheels_heading = 0
        self.absolute_velocity = 0
        self.path = path
        self.path_dist = 0
        self.cos_path_ang_diff = 1
        self.sin_path_ang_diff = 0
        self.path_long_dist = 0
        self.next_path_dist = 0
        self.cos_next_path_ang_diff = 1
        self.sin_next_path_ang_diff = 0
        self.next_path_long_dist = 0
        self.next_next_path_dist = 0
        self.cos_next_next_path_ang_diff = 1
        self.sin_next_next_path_ang_diff = 0
        self.next_next_path_long_dist = 0
        self.next_next_next_path_dist = 0
        self.cos_next_next_next_path_ang_diff = 1
        self.sin_next_next_next_path_ang_diff = 0
        self.next_next_next_path_long_dist = 0
        

    @classmethod
    def make_on_lane(cls, road: Road, lane_index: LaneIndex, longitudinal: float, speed: float = 0) -> "Vehicle":
        """
        Create a vehicle on a given lane at a longitudinal position.

        :param road: the road where the vehicle is driving
        :param lane_index: index of the lane where the vehicle is located
        :param longitudinal: longitudinal position along the lane
        :param speed: initial speed in [m/s]
        :return: A vehicle with at the specified position
        """
        lane = road.network.get_lane(lane_index)
        if speed is None:
            speed = lane.speed_limit
        return cls(road, lane.position(longitudinal, 0), lane.heading_at(longitudinal), speed)

    @classmethod
    def create_random(cls, road: Road,
                      speed: float = None,
                      lane_from: Optional[str] = None,
                      lane_to: Optional[str] = None,
                      lane_id: Optional[int] = None,
                      spacing: float = 1) \
            -> "Vehicle":
        """
        Create a random vehicle on the road.

        The lane and /or speed are chosen randomly, while longitudinal position is chosen behind the last
        vehicle in the road with density based on the number of lanes.

        :param road: the road where the vehicle is driving
        :param speed: initial speed in [m/s]. If None, will be chosen randomly
        :param lane_from: start node of the lane to spawn in
        :param lane_to: end node of the lane to spawn in
        :param lane_id: id of the lane to spawn in
        :param spacing: ratio of spacing to the front vehicle, 1 being the default
        :return: A vehicle with random position and/or speed
        """
        _from = lane_from or road.np_random.choice(list(road.network.graph.keys()))
        _to = lane_to or road.np_random.choice(list(road.network.graph[_from].keys()))
        _id = lane_id if lane_id is not None else road.np_random.choice(len(road.network.graph[_from][_to]))
        lane = road.network.get_lane((_from, _to, _id))
        if speed is None:
            if lane.speed_limit is not None:
                speed = road.np_random.uniform(0.7*lane.speed_limit, lane.speed_limit)
            else:
                speed = road.np_random.uniform(Vehicle.DEFAULT_SPEEDS[0], Vehicle.DEFAULT_SPEEDS[1])
        default_spacing = 15+1.2*speed
        offset = spacing * default_spacing * np.exp(-5 / 40 * len(road.network.graph[_from][_to]))
        x0 = np.max([lane.local_coordinates(v.position)[0] for v in road.vehicles]) \
            if len(road.vehicles) else 3*offset
        x0 += offset * road.np_random.uniform(0.9, 1.1)
        v = cls(road, lane.position(x0, 0), lane.heading_at(x0), speed)
        return v

    @classmethod
    def create_from(cls, vehicle: "Vehicle") -> "Vehicle":
        """
        Create a new vehicle from an existing one.

        Only the vehicle dynamics are copied, other properties are default.

        :param vehicle: a vehicle
        :return: a new vehicle at the same dynamical state
        """
        v = cls(vehicle.road, vehicle.position, vehicle.heading, vehicle.speed)
        return v

    def act(self, action: Union[dict, str] = None) -> None:
        """
        Store an action to be repeated.

        :param action: the input action
        """
        if action:
            self.action = action

    def step(self, dt: float) -> None:
        """
        Propagate the vehicle state given its actions.

        Integrate a modified bicycle model with a 1st-order response on the steering wheel dynamics.
        If the vehicle is crashed, the actions are overridden with erratic steering and breaking until complete stop.
        The vehicle's current lane is updated.

        :param dt: timestep of integration of the model [s]
        """
        self.clip_actions()
        delta_f = self.action['steering']
        beta = np.arctan(1 / 2 * np.tan(delta_f))
        v = self.speed * np.array([np.cos(self.heading + beta),
                                   np.sin(self.heading + beta)])
        self.position += v * dt
        if self.impact is not None:
            self.position += self.impact
            self.crashed = True
            self.impact = None
        self.heading += self.speed * np.sin(beta) / (self.LENGTH / 2) * dt
        self.heading = self.heading_modulus(self.heading)
        self.speed += self.action['acceleration'] * dt
        self.on_state_update()

    def get_distance_to_closest_path_points_to_vehicle(self, num_points : int, path : list, track_rear):
        taking_out_used_points = False
        vehicle_position = self.position
        if track_rear == 1:
            half_vehicle_length = self.LENGTH / 2
            vehicle_rear_position = [self.position[0] + half_vehicle_length * math.cos(self.heading + math.pi), self.position[1] + half_vehicle_length * math.sin(self.heading + math.pi)]
            vehicle_position = tuple(vehicle_rear_position)
        
        if path == None:
            return 0

        path_points_distance = self.get_path_points_distance(vehicle_position, path)
        
        closest_points, index_of_next_point_in_path = self.get_closest_points(num_points, path_points_distance, path)

        #print("Closest point index: " + str(index_of_next_point_in_path))
        point3 = vehicle_position


        #UPDATE CURRENT PATH POINT VALUES
        point1, point2, path_angle1, path_angle2 = self.get_closest_path_points_and_angles(index_of_next_point_in_path, path)

        abs_lat_dist_to_path = self.get_lateral_distance_from_points(point1, point2, point3)

        if self.isLeft(point1, point2, point3):
            self.path_dist = abs_lat_dist_to_path
        else:
            self.path_dist = -abs_lat_dist_to_path

        path_angle = (path_angle1 + path_angle2)/2
        path_angle_modulus = self.heading_modulus(path_angle - self.heading)
        self.cos_path_ang_diff = math.cos(path_angle_modulus)
        self.sin_path_ang_diff = math.sin(path_angle_modulus)

        #print(str(self.cos_path_ang_diff) + " // " + str(self.sin_path_ang_diff))

        self.path_long_dist = self.get_longitudinal_distance_to_following_point_to_the_closest_path_point(index_of_next_point_in_path, path)


        #UPDATE NEXT PATH POINT VALUES
        next_point1, next_point2, next_path_angle1, next_path_angle2 = self.get_closest_path_points_and_angles(index_of_next_point_in_path + 1, path)

        next_abs_lat_dist_to_path = self.get_lateral_distance_from_points(next_point1, next_point2, point3)

        if self.isLeft(next_point1, next_point2, point3):
            self.next_path_dist = next_abs_lat_dist_to_path
        else:
            self.next_path_dist = -next_abs_lat_dist_to_path
        
        next_path_angle = (next_path_angle1 + next_path_angle2)/2
        next_path_angle_modulus = self.heading_modulus(next_path_angle - self.heading)
        self.cos_next_path_ang_diff = math.cos(next_path_angle_modulus)
        self.sin_next_path_ang_diff = math.sin(next_path_angle_modulus)

        self.next_path_long_dist = self.get_longitudinal_distance_to_following_point_to_the_closest_path_point(index_of_next_point_in_path + 1, path)

        #UPDATE NEXT NEXT PATH POINT VALUES
        next_next_point1, next_next_point2, next_next_path_angle1, next_next_path_angle2 = self.get_closest_path_points_and_angles(index_of_next_point_in_path + 2, path)

        next_next_abs_lat_dist_to_path = self.get_lateral_distance_from_points(next_next_point1, next_next_point2, point3)

        if self.isLeft(next_next_point1, next_next_point2, point3):
            self.next_next_path_dist = next_next_abs_lat_dist_to_path
        else:
            self.next_next_path_dist = -next_next_abs_lat_dist_to_path
        
        next_next_path_angle = (next_next_path_angle1 + next_next_path_angle2)/2
        next_next_path_angle_modulus = self.heading_modulus(next_next_path_angle - self.heading)
        self.cos_next_next_path_ang_diff = math.cos(next_next_path_angle_modulus)
        self.sin_next_next_path_ang_diff = math.sin(next_next_path_angle_modulus)

        self.next_next_path_long_dist = self.get_longitudinal_distance_to_following_point_to_the_closest_path_point(index_of_next_point_in_path + 2, path)

        #UPDATE NEXT NEXT NEXT PATH POINT VALUES
        next_next_next_point1, next_next_next_point2, next_next_next_path_angle1, next_next_next_path_angle2 = self.get_closest_path_points_and_angles(index_of_next_point_in_path + 3, path)

        next_next_next_abs_lat_dist_to_path = self.get_lateral_distance_from_points(next_next_next_point1, next_next_next_point2, point3)

        if self.isLeft(next_next_next_point1, next_next_next_point2, point3):
            self.next_next_next_path_dist = next_next_next_abs_lat_dist_to_path
        else:
            self.next_next_next_path_dist = -next_next_next_abs_lat_dist_to_path
        
        next_next_next_path_angle = (next_next_next_path_angle1 + next_next_next_path_angle2)/2
        next_next_next_path_angle_modulus = self.heading_modulus(next_next_next_path_angle - self.heading)
        self.cos_next_next_next_path_ang_diff = math.cos(next_next_next_path_angle_modulus)
        self.sin_next_next_next_path_ang_diff = math.sin(next_next_next_path_angle_modulus)

        self.next_next_next_path_long_dist = self.get_longitudinal_distance_to_following_point_to_the_closest_path_point(index_of_next_point_in_path + 3, path)

        self.current_acceleration = self.action['acceleration']
        self.front_wheels_heading = self.action['steering']
        self.absolute_velocity = self.get_vehicle_velocity()
        #print(self.action['acceleration'])
        #print(self.get_vehicle_velocity())
        #print(self.action['steering'])
        #print("ANG_DIFF = " + str(self.path_ang_diff) + " // NEXT_ANG_DIFF = " + str(self.next_path_ang_diff) )

        if taking_out_used_points and len(closest_points) > 0 and abs_lat_dist_to_path < 0.25 and abs(path_angle_modulus) < 0.1 and  abs(self.path_long_dist) < 1.5:
            #print(len(path) - index_of_next_point_in_path)
            path = path[-(len(path) - index_of_next_point_in_path):]
        
        #print(len(path))
        return abs_lat_dist_to_path, abs(path_angle_modulus), self.path_long_dist, index_of_next_point_in_path, path
    
    def get_path_points_distance(self, vehicle_position, path):
        only_consider_next_points = False
        num_of_next_points_to_consider = 10

        path_points_distance = []
        if only_consider_next_points and len(path) > num_of_next_points_to_consider:
            path_to_observe = path[:num_of_next_points_to_consider]
        else:
            path_to_observe = path
        for point in path_to_observe:
            path_points_distance.append([point, self.calculate_points_distance_considering_angle([point[0], point[1]], point[2], vehicle_position, self.heading)])
        path_points_distance = sorted(path_points_distance, key=itemgetter(1))

        return path_points_distance

    def get_closest_points(self, num_points, path_points_distance, path):
        closest_points = []
        if len(path_points_distance) >= num_points:
            closest_points = path_points_distance[:num_points]
            index_of_next_point_in_path = path.index(closest_points[0][0])
        else:
            closest_points = path_points_distance[-1]
            index_of_next_point_in_path = path.index(closest_points[0])
        return closest_points, index_of_next_point_in_path

    
    def get_longitudinal_distance_to_following_point_to_the_closest_path_point(self, index_of_next_point_in_path, path):
        if index_of_next_point_in_path + 2 > len(path):
            point3 = [path[-1][0], path[-1][1]]
        else:
            point3 = [path[index_of_next_point_in_path + 1][0], path[index_of_next_point_in_path + 1][1]]

        point1, point2 = self.get_vehicle_rear_wheels_positions()

        absolute_distance = self.get_lateral_distance_from_points(point1, point2, point3)
        if self.isLeft(point1, point2, point3):
            distance = absolute_distance
        else:
            distance = -absolute_distance
        return distance
        #print(self.path_long_dist)


        
    def get_vehicle_rear_wheels_positions(self):
        half_vehicle_length = self.LENGTH / 2
        half_vehicle_width = self.WIDTH / 2
        vehicle_wheels_positions = []

        #vehicle_front_position = [self.position[0] + half_vehicle_length * math.cos(self.heading), self.position[1] + half_vehicle_length * math.sin(self.heading)]
        vehicle_rear_position = [self.position[0] + half_vehicle_length * math.cos(self.heading + math.pi), self.position[1] + half_vehicle_length * math.sin(self.heading + math.pi)]

        #front_rigth_wheel_position = [vehicle_front_position[0] + half_vehicle_width * math.cos(vehicle.heading + math.pi/2), vehicle_front_position[1] + half_vehicle_width * math.sin(vehicle.heading + math.pi/2)]
        #front_left_wheel_position = [vehicle_front_position[0] - half_vehicle_width * math.cos(vehicle.heading + math.pi/2), vehicle_front_position[1] - half_vehicle_width * math.sin(vehicle.heading + math.pi/2)]
        rear_rigth_wheel_position = [vehicle_rear_position[0] + half_vehicle_width * math.cos(self.heading + math.pi/2), vehicle_rear_position[1] + half_vehicle_width * math.sin(self.heading + math.pi/2)]
        rear_left_wheel_position = [vehicle_rear_position[0] - half_vehicle_width * math.cos(self.heading + math.pi/2), vehicle_rear_position[1] - half_vehicle_width * math.sin(self.heading + math.pi/2)]

        #vehicle_wheels_positions.append(front_rigth_wheel_position)
        #vehicle_wheels_positions.append(front_left_wheel_position)
        vehicle_wheels_positions.append(rear_rigth_wheel_position)
        vehicle_wheels_positions.append(rear_left_wheel_position)

        return rear_rigth_wheel_position, rear_left_wheel_position


    def get_closest_path_points_and_angles(self, index_of_next_point_in_path, path):
        path_angle1 = 0
        path_angle2 = 0
        if index_of_next_point_in_path + 1 < len(path):
            point1 = [path[index_of_next_point_in_path][0], path[index_of_next_point_in_path][1]]
            point2 = [path[index_of_next_point_in_path + 1][0], path[index_of_next_point_in_path + 1][1]]
            path_angle1 = path[index_of_next_point_in_path][2]
            path_angle2 = path[index_of_next_point_in_path + 1][2]
        else:
            point1 = [path[len(path) - 2][0], path[len(path) - 2][1]]
            point2 = [path[len(path) - 1][0], path[len(path) - 1][1]]
            path_angle1 = path[len(path) - 2][2]
            path_angle2 = path[len(path) - 1][2]
            #print(str(self.isLeft([path[index_of_next_point_in_path][0], path[index_of_next_point_in_path][1]], [path[index_of_next_point_in_path + 1][0], path[index_of_next_point_in_path + 1][1]], self.position)) + " / " + str(abs_lat_dist_to_path))
        return point1, point2, path_angle1, path_angle2
    
    def get_lateral_distance_from_points(self, point1, point2, point3):#Point 1 and 2 make straight line, point3's distance to straight is measured, negative to the left, positive to the right, but returns absolute value
        
        #abs_lat_dist_to_path = linalg.norm(np.cross(np.array(point2)-np.array(point1), np.array(point1)-np.array(point3)))/linalg.norm(np.array(point2)-np.array(point1))
        denominator = np.sqrt(np.square(point2[0]-point1[0]) + np.square(point2[1]-point1[1]))
        numerator = abs((point2[0]-point1[0])*(point1[1]-point3[1]) - (point1[0]-point3[0])*(point2[1]-point1[1]))
        if denominator == 0:
            #print("DENOMINATOR = 0 // NUMERATOR = " + str(numerator))
            abs_lat_dist_to_path = 0
        else:
            abs_lat_dist_to_path = numerator / denominator
        
        
        #print(self.path_dist)
        return abs_lat_dist_to_path

    def calculate_heading_from_two_points(self, point1, point2):
        position_dif = [point2[0] - point1[0], point2[1] - point1[1]]

        position_dif_angle = math.atan2(position_dif[1], position_dif[0])

        return position_dif_angle

    def heading_modulus(self, heading):
        while heading > math.pi:
            heading -= 2*math.pi
        while heading < -math.pi:
            heading += 2*math.pi
        return heading

    def isLeft(self, point1: list, point2: list, point3) -> bool:
        return ((point2[0] - point1[0])*(point3[1] - point1[1]) - (point2[1] - point1[1])*(point3[0] - point1[0])) > 0

    def calculate_points_distance(self, point1, point2):
        x_dif = point1[0] - point2[0]
        y_dif = point1[1] - point2[1]

        return math.sqrt(x_dif*x_dif + y_dif*y_dif)

    def calculate_points_distance_considering_angle(self, point1, heading1, point2, heading2):
        x_dif = point1[0] - point2[0]
        y_dif = point1[1] - point2[1]

        abs_distance = math.sqrt(x_dif*x_dif + y_dif*y_dif)
        angle_diff = self.heading_modulus(heading1 - heading2)

        return abs_distance * (1 + math.sqrt(abs(angle_diff))) + (2.5 * math.sqrt(abs(angle_diff)))

    def get_vehicle_velocity(self):
        return math.sqrt(self.velocity[0]**2 + self.velocity[1]**2)

    def clip_actions(self) -> None:
        if self.crashed:
            self.action['steering'] = 0
            self.action['acceleration'] = -1.0*self.speed
        self.action['steering'] = float(self.action['steering'])
        self.action['acceleration'] = float(self.action['acceleration'])
        if self.speed > self.MAX_SPEED:
            self.action['acceleration'] = min(self.action['acceleration'], 1.0 * (self.MAX_SPEED - self.speed))
        elif self.speed < -self.MAX_SPEED:
            self.action['acceleration'] = max(self.action['acceleration'], 1.0 * (self.MAX_SPEED - self.speed))

    def on_state_update(self) -> None:
        if self.road:
            self.lane_index = self.road.network.get_closest_lane_index(self.position, self.heading)
            self.lane = self.road.network.get_lane(self.lane_index)
            if self.road.record_history:
                self.history.appendleft(self.create_from(self))

    def check_collision(self, other: 'RoadObject', dt: float = 0) -> None:
        """
        Check for collision with another vehicle.

        :param other: the other vehicle or object
        :param dt: timestep to check for future collisions (at constant velocity)
        """
        if other is self:
            return

        if isinstance(other, Vehicle):
            if not self.COLLISIONS_ENABLED or not other.COLLISIONS_ENABLED:
                return
            intersecting, will_intersect, transition = self._is_colliding(other, dt)
            if will_intersect:
                self.impact = transition / 2
                other.impact = -transition / 2
            if intersecting:
                self.crashed = other.crashed = True
        elif isinstance(other, Obstacle):
            if not self.COLLISIONS_ENABLED:
                return
            intersecting, will_intersect, transition = self._is_colliding(other, dt)
            if will_intersect:
                self.impact = transition
            if intersecting:
                self.crashed = other.hit = True
        elif isinstance(other, Landmark):
            intersecting, will_intersect, transition = self._is_colliding(other, dt)
            if intersecting:
                other.hit = True

    def _is_colliding(self, other, dt):
        # Fast spherical pre-check
        if np.linalg.norm(other.position - self.position) > self.LENGTH + self.speed * dt:
            return False, False, np.zeros(2,)
        # Accurate rectangular check
        return utils.are_polygons_intersecting(self.polygon(), other.polygon(), self.velocity * dt, other.velocity * dt)

    @property
    def velocity(self) -> np.ndarray:
        return self.speed * self.direction  # TODO: slip angle beta should be used here

    @property
    def destination(self) -> np.ndarray:
        if getattr(self, "route", None):
            last_lane_index = self.route[-1]
            last_lane_index = last_lane_index if last_lane_index[-1] is not None else (*last_lane_index[:-1], 0)
            last_lane = self.road.network.get_lane(last_lane_index)
            return last_lane.position(last_lane.length, 0)
        else:
            return self.position

    @property
    def destination_direction(self) -> np.ndarray:
        if (self.destination != self.position).any():
            return (self.destination - self.position) / np.linalg.norm(self.destination - self.position)
        else:
            return np.zeros((2,))

    def to_dict(self, origin_vehicle: "Vehicle" = None, observe_intentions: bool = True) -> dict:
        d = {
            'presence': 1,
            'x': self.position[0],
            'y': self.position[1],
            'vx': self.velocity[0],
            'vy': self.velocity[1],
            'heading': self.heading,
            'cos_h': self.direction[0],
            'sin_h': self.direction[1],
            'cos_d': self.destination_direction[0],
            'sin_d': self.destination_direction[1],
            'current_acceleration': self.current_acceleration,
            'front_wheels_heading': self.front_wheels_heading,
            'absolute_velocity': self.absolute_velocity,
            'path_dist': self.path_dist,
            'cos_path_ang_diff': self.cos_path_ang_diff,
            'sin_path_ang_diff': self.sin_path_ang_diff,
            'path_long_dist': self.path_long_dist,
            'next_path_dist': self.next_path_dist,
            'cos_next_path_ang_diff': self.cos_next_path_ang_diff,
            'sin_next_path_ang_diff': self.sin_next_path_ang_diff,
            'next_path_long_dist': self.next_path_long_dist,
            'next_next_path_dist': self.next_next_path_dist,
            'cos_next_next_path_ang_diff': self.cos_next_next_path_ang_diff,
            'sin_next_next_path_ang_diff': self.sin_next_next_path_ang_diff,
            'next_next_path_long_dist': self.next_next_path_long_dist,
            'next_next_next_path_dist': self.next_next_next_path_dist,
            'cos_next_next_next_path_ang_diff': self.cos_next_next_next_path_ang_diff,
            'sin_next_next_next_path_ang_diff': self.sin_next_next_next_path_ang_diff,
            'next_next_next_path_long_dist': self.next_next_next_path_long_dist
        }
        if not observe_intentions:
            d["cos_d"] = d["sin_d"] = 0
        if origin_vehicle:
            origin_dict = origin_vehicle.to_dict()
            for key in ['x', 'y', 'vx', 'vy']:
                d[key] -= origin_dict[key]
        return d

    def __str__(self):
        return "{} #{}: {}".format(self.__class__.__name__, id(self) % 1000, self.position)

    def __repr__(self):
        return self.__str__()
