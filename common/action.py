from typing import Dict, List, Optional, Tuple, Union
from math import *

import numpy as np

import carla

from gym import spaces

import common.utils as utils
from controller import Controller

Action = Union[int, np.ndarray]

class ActionType(object):

    """A type of action specifies its definition space, and how actions are executed in the environment"""

    def __init__(self, world: carla.World, **kwargs) -> None:
        self.world = world
        self.__controlled_vehicle = None

    def space(self) -> spaces.Space:
        """The action space."""
        raise NotImplementedError

    def act(self, action: Action) -> None:
        """
        Execute the action on the ego-vehicle.

        :param action: the action to execute
        """
        raise NotImplementedError

    @property
    def controlled_vehicle(self):
        """The vehicle acted upon.

        If not set, the first controlled vehicle is used by default."""
        return self.__controlled_vehicle

    @controlled_vehicle.setter
    def controlled_vehicle(self, vehicle):
        self.__controlled_vehicle = vehicle

class ParameterTuningAction(ActionType):
    """Kp parameter tuning range: [0, 10.0]."""
    Kp_RANGE = (0.0, 10.0)

    """Ki parameter tuning range: [0, 10.0]."""
    Ki_RANGE = (0.0, 10.0)

    """Kd parameter tuning range: [0, 10.0]."""
    Kd_RANGE = (0.0, 10.0)

    """Desired speed parameter tuning range: [0.0, 10.0], in m/s."""
    Desired_speed_RANGE = (0.0, 10.0)

    """Look ahead distance parameter tuning range: [20.0, 50.0], in meters."""
    Look_ahead_distance_RANGE = (20.0, 50.0)

    def __init__(
            self, 
            world: carla.World,
            kp_range: Optional[Tuple[float, float]] = None,
            ki_range: Optional[Tuple[float, float]] = None,
            kd_range: Optional[Tuple[float, float]] = None,
            desired_speed_range: Optional[Tuple[float, float]] = None,
            look_ahead_distance_range: Optional[Tuple[float, float]] = None,
            **kwargs) -> None:
        """
        Create a parameter tuning action space.

        :param world: the CARLA Simulation
        :param kp_range: the range of the proportional gain
        :param ki_range: the range of the integral gain
        :param kd_range: the range of the derivative gain
        :param desired_speed_range: the range of the desired speed
        :param look_ahead_distance_range: the range of the look ahead distance
        """
        super().__init__(world)
        self._waypoints = None
        self._debug_waypoints = None
        self._controller = None
        self.step = 0

        self.kp_range = kp_range if kp_range else self.Kp_RANGE
        self.ki_range = ki_range if ki_range else self.Ki_RANGE
        self.kd_range = kd_range if kd_range else self.Kd_RANGE
        self.desired_speed_range = desired_speed_range if desired_speed_range else self.Desired_speed_RANGE
        self.look_ahead_distance_range = look_ahead_distance_range if look_ahead_distance_range else self.Look_ahead_distance_RANGE

        self.size = 5
        self.last_action = np.zeros(self.size)

    @property
    def waypoints(self):
        return self._waypoints
    
    @waypoints.setter
    def waypoints(self, waypoints):
        self._waypoints = waypoints

    @property
    def debug_waypoints(self):
        return self._debug_waypoints
    
    @debug_waypoints.setter
    def debug_waypoints(self, debug_waypoints):
        self._debug_waypoints = debug_waypoints

    @property
    def controller(self):
        return self._controller
    
    @controller.setter
    def controller(self, controller):
        self._controller = controller

    def space(self) -> spaces.Box:
        return spaces.Box(-1.0, 1.0, shape=(self.size, ), dtype=np.float32)
    
    def get_current_velocity(self, vehicle: carla.Vehicle) -> float:
        velocity = vehicle.get_velocity()
        vx = velocity.x
        vy = velocity.y

        return sqrt(vx**2 + vy**2)
    
    def get_vehicle_info(self) -> Tuple[float, float, float, carla.Timestamp]:
        x = self.controlled_vehicle.get_location().x
        y = self.controlled_vehicle.get_location().y
        yaw = self.controlled_vehicle.get_transform().rotation.yaw
        speed = self.get_current_velocity(self.controlled_vehicle)

        self.world.tick()
        timestamp = self.world.get_snapshot().timestamp.elapsed_seconds

        return x, y, yaw, speed, timestamp

    def calculate_distance(self, x1: float, y1: float, x2: float, y2: float) -> float:
        return sqrt((x2 - x1) ** 2 + (y2 - y1)  ** 2)

    def get_closest_waypoint(self) -> int:
        THRESHOLD = 3.0

        min_distance = float('inf')
        closest_index = -1

        current_x = self.controlled_vehicle.get_location().x
        current_y = self.controlled_vehicle.get_location().y
        current_yaw = self.controlled_vehicle.get_transform().rotation.yaw

        # Calculate vehicle's direction vector
        vehicle_direction = np.array([np.cos(np.radians(current_yaw)), np.sin(np.radians(current_yaw))])

        for index, waypoint in enumerate(self.waypoints):
            waypoint_x, waypoint_y, _ = waypoint
            distance = self.calculate_distance(current_x, current_y, waypoint_x, waypoint_y)
            
            # Calculate relative position vector between waypoint and vehicle
            waypoint_direction = np.array([waypoint_x - current_x, waypoint_y - current_y])
            
            # Check the relative position vector projected onto the vehicle's direction vector
            dot_product = np.dot(vehicle_direction, waypoint_direction)

            # If the distance is greater than THRESHOLD and the waypoint is in front of the vehicle, find the index of the closest waypoint
            if distance >= THRESHOLD and distance < min_distance and dot_product > 0:
                min_distance = distance
                closest_index = index
        
        return closest_index

    # Find the index of the last waypoint within the Look Ahead Distance
    def get_last_index_within_look_ahead_distance(self, look_ahead_distance: float) -> int:
        last_index = -1

        current_x = self.controlled_vehicle.get_location().x
        current_y = self.controlled_vehicle.get_location().y

        for index, waypoint in enumerate(self.waypoints):
            waypoint_x, waypoint_y, _ = waypoint
            distance = self.calculate_distance(current_x, current_y, waypoint_x, waypoint_y)
            
            if distance <= look_ahead_distance:
                last_index = index
        
        return last_index
    
    def send_control_command(self, vehicle: carla.Vehicle, throttle: float, steer: float, brake: float) -> None:
        steer = np.fmax(np.fmin(steer, 1.0), -1.0)
        throttle = np.fmax(np.fmin(throttle, 1.0), 0)
        brake = np.fmax(np.fmin(brake, 1.0), 0)
        vehicle.apply_control(carla.VehicleControl(throttle=throttle, steer=steer, brake=brake))

    def get_action(self, action: np.ndarray) -> np.ndarray:
        return {
            "Kp": utils.lmap(action[0], (-1, 1), self.kp_range),
            "Ki": utils.lmap(action[1], (-1, 1), self.ki_range),
            "Kd": utils.lmap(action[2], (-1, 1), self.kd_range),
            "Desired_speed": utils.lmap(action[3], (-1, 1), self.desired_speed_range),
            "Look_ahead_distance": utils.lmap(action[4], (-1, 1), self.look_ahead_distance_range)
        }
    
    def act(self, action: np.ndarray) -> None:
        action_dict = self.get_action(action)

        current_x, current_y, current_yaw, current_speed, current_timestamp = self.get_vehicle_info()

        closest_idx = self.get_closest_waypoint()

        look_ahead_idx = self.get_last_index_within_look_ahead_distance(action_dict['Look_ahead_distance'])

        ref_waypoints = self.waypoints[closest_idx:look_ahead_idx]
        ref_debug_waypoints = self.debug_waypoints[closest_idx:look_ahead_idx]

        # TODO: Needs to be modified later (The method does not return anything)
        if len(ref_waypoints) < 2:
            return True
        
        # Display waypoints within the look ahead distance that the current vehicle refers to on the map
        for waypoint in ref_debug_waypoints:
            location = carla.Location(x=waypoint[0], y=waypoint[1], z=waypoint[2])
            self.world.debug.draw_point(location, size=0.1, color=carla.Color(0, 130, 125), life_time=0.2)
        
        self.controller.update_waypoints(ref_waypoints)
        self.controller.update_values(current_x, current_y, current_yaw, current_speed, current_timestamp, self.step)
        self.controller.update_parameters(K_p=action_dict["Kp"], K_i=action_dict["Ki"], K_d=action_dict["Kd"], desired_speed=action_dict["Desired_speed"])
        self.controller.update_controls()

        cmd_throttle, cmd_steer, cmd_brake = self.controller.get_commands()
        self.send_control_command(self.controlled_vehicle, cmd_throttle, cmd_steer, cmd_brake)

        self.last_action = action
        self.step += 1

class MultiAgentAction(ActionType):       
    def __init__(self, world: carla.World, action_config: Dict, **kwargs) -> None:
        super().__init__(world)
        if 'controlled_vehicles' in kwargs and 'total_waypoints' in kwargs:
            self.controlled_vehicles = kwargs['controlled_vehicles']
            self.total_waypoints = kwargs['total_waypoints']
            
            self.action_config = action_config
            self.agents_action_types = []

            for vehicle, total_waypoint in zip(self.controlled_vehicles, self.total_waypoints):
                waypoints = total_waypoint[0]
                debug_waypoints = total_waypoint[1]
                
                action_type = action_factory(world, self.action_config)
                action_type.waypoints = waypoints
                action_type.debug_waypoints = debug_waypoints
                action_type.controlled_vehicle = vehicle
                action_type.controller = Controller(waypoints)

                self.agents_action_types.append(action_type)

    def space(self) -> spaces.Space:
        return spaces.Tuple(
            [action_type.space() for action_type in self.agents_action_types]
        )
    
    def act(self, action: Action) -> None:
        assert isinstance(action, tuple)
        for agent_action, action_type in zip(action, self.agents_action_types):
            action_type.act(agent_action)

def action_factory(world: carla.World, config: Dict) -> ActionType:
    if config["type"] == 'ParameterTuningAction':
        return ParameterTuningAction(world, **config)
    elif config["type"] == 'MultiAgentAction':
        return MultiAgentAction(world, **config)
    else:
        raise ValueError("Unknown action type {}".format(config["type"]))