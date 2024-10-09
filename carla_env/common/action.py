from typing import Optional, Tuple, Union

import numpy as np

import gym
from gym import spaces

import carla
from agents.navigation.controller import VehiclePIDController, PIDLongitudinalController, PIDLateralController

from carla_env.common.utils import lmap

Action = Union[int, np.ndarray]

class ActionType(object):
    """A type of action specifies its definition space, and how actions are executed in the environment"""

    def __init__(self, env: gym.Env,**kwargs) -> None:
        self.env = env
        self.__controlled_vehicle = None

    def space(self) -> spaces.Space:
        """The action space."""
        raise NotImplementedError
    
    def act(self, action: Action) -> None:
        """
        Execute the action on the ego-vehicle.

        :param action: the action to exectue
        """
        raise NotImplementedError
    
    @property
    def controlled_vehicle(self):
        """
        The vehicle action upon.
        
        If not set, the first controlled vehicle is used by default.
        """
        return self.__controlled_vehicle or self.env.vehicle
    
    @controlled_vehicle.setter
    def controlled_vehicle(self, vehicle):
        self.__controlled_vehicle = vehicle

class ContinuousAction(ActionType):
    """
    An continuous action space for throttle and steering angle.

    If both throttle and streeing are enabled, they are set in this order: [throttle, steering]

    The space intervals are always [-1, 1], but are mapped to throttle/steering intervals through configurations.
    """

    THROTTLE_RANGE = [0.0, 1.0]
    STEER_RANGE = [-1.0, 1.0]

    def __init__(
            self,
            env: gym.Env,
            throttle_range: Optional[Tuple[float, float]] = None,
            steer_range: Optional[Tuple[float, float]] = None,
            longitudinal: bool = True,
            lateral: bool = True,
            **kwargs: dict,
    ) -> None:
        """
        Create a continuous action space.

        :param env: the environment
        :param throttle_range: the range of throttle values
        :param steer_range: the range of steering values
        :param longitudinal: enable throttle control
        :param lateral: enable steering control
        """

        super().__init__(env)
        
        self.throttle_range = throttle_range if throttle_range else self.THROTTLE_RANGE
        self.steer_range = steer_range if steer_range else self.STEER_RANGE
        self.longitudinal = longitudinal
        self.lateral = lateral

        if not self.longitudinal and not self.lateral:
            raise ValueError("Either longitudinal and/or lateral control must be enabled")
        
        self.size = 2 if self.longitudinal and self.lateral else 1
        self.last_action = np.zeros(self.size)

        self.target_speed = 0
        self.waypoint_idx = 0
        self.is_arrived = False

    def space(self) -> spaces.Box:
        return spaces.Box(-1.0, 1.0, shape=(self.size,), dtype=np.float32)
    
    def get_action(self, action: np.ndarray):
        if self.longitudinal and self.lateral:
            return {
                "throttle": lmap(action[0], [-1, 1], self.throttle_range),
                "steer": lmap(action[1], [-1, 1], self.steer_range),
            }
        
        elif self.longitudinal:
            return {"throttle": lmap(action[0], [-1, 1], self.throttle_range)}
        
        elif self.lateral:
            return {"steer": lmap(action[0], [-1, 1], self.steer_range)}
    
    def setup_PID(self) -> PIDLongitudinalController:
        """
        Setup the PID controller for the vehicle.
        If longitudinal and lateral control are enabled, do not use any controller.
        If only longitudinal is enabled, lateral control is automatically executed.
        If only lateral is enabled, longitudinal control is automatically executed.
        """

        if self.longitudinal and self.lateral:
            pass

        elif self.longitudinal:
            self.controller = PIDLateralController(self.controlled_vehicle, K_P=1.95, K_I=0.2, K_D=0.07, dt=1.0 / 10.0)

        elif self.lateral:
            self.controller = PIDLongitudinalController(self.controlled_vehicle, K_P=1.0, K_I=0.0, K_D=0.75, dt=1.0 / 10.0)
            

    @property
    def controlled_vehicle(self):
        """
        The vehicle action upon.
        
        If not set, the first controlled vehicle is used by default.
        """
        return self._controlled_vehicle or self.env.vehicle
    
    @controlled_vehicle.setter
    def controlled_vehicle(self, vehicle):
        self._controlled_vehicle = vehicle
        self.setup_PID()

    @property
    def waypoints(self):
        return self._waypoints
    
    @waypoints.setter
    def waypoints(self, waypoints):
        self._waypoints = waypoints

    def calculate_distance(self, loc1: carla.Location, loc2: carla.Location) -> float:
        return loc1.distance(loc2)
    
    def act(self, action: np.ndarray) -> None:
        """
        If both longitudinal and lateral control are enabled, the action is [throttle, steer].
        If only longitudinal control is enabled, the action is throttle. lateral control is automatically executed.
        If only lateral control is enabled, the action is steer. longitudinal control is automatically executed.
        """
        
        if self.longitudinal and self.lateral:
            action_dict = self.get_action(action)
            self.controlled_vehicle.apply_control(carla.VehicleControl(throttle=action_dict['throttle'], steer=action_dict['steer']))

        elif self.longitudinal:
            action_dict = self.get_action(action)

            distance = self.calculate_distance(self.controlled_vehicle.get_location(), self.waypoints[self.waypoint_idx].transform.location)
            steer = self.controller.run_step(self.waypoints[self.waypoint_idx])
            control = carla.VehicleControl(throttle=action_dict['throttle'], steer=steer)
            self.controlled_vehicle.apply_control(control)
    
            if self.waypoint_idx == (len(self.waypoints) - 1):
                self.controlled_vehicle.apply_control(carla.VehicleControl(brake=1.0, steer=0.0))
                self.is_arrived = True

            if distance < 3.5:
                self.waypoint_idx += 1
        
        elif self.lateral:
            action_dict = self.get_action(action)
            throttle = self.controller.run_step(self.target_speed)
            control = carla.VehicleControl(throttle=throttle, steer=action_dict['steer'])
            self.controlled_vehicle.apply_control(control)

class DiscreteMetaAction(ActionType):
    """
    An discrete action space of meta-actions: lane changes, and cruise control set-point.
    """

    ACTIONS_ALL = {0: "LANE_LEFT", 1: "IDLE", 2: "LANE_RIGHT", 3: "FASTER", 4: "SLOWER"}
    """A mapping of action indexes to labels."""

    ACTIONS_LONGI = {0: "SLOWER", 1: "IDLE", 2: "FASTER"}
    """A mapping of longitudinal action indexes to labels."""

    ACTIONS_LAT = {0: "LANE_LEFT", 1: "IDLE", 2: "LANE_RIGHT"}
    """A mapping of lateral action indexes to labels."""

    def __init__(
            self, 
            env: gym.Env,
            longitudinal: bool = True,
            lateral: bool = True,
            **kwargs) -> None:
        """
        Create a parameter tuning action space.

        :param world: the CARLA Simulation
        :param  longitudinal: include longitudinal actions
        :param  lateral: include lateral actions
        """
        super().__init__(env)
        self._waypoints = None
        self._debug_waypoints = None
        self._controller = None

        # TODO: Needs to be modified later
        self.target_speed = 0

        self.step = 0

        self.longitudinal = longitudinal
        self.lateral = lateral
        self.actions = (
            self.ACTIONS_ALL 
            if longitudinal and lateral 
            else self.ACTIONS_LONGI
            if longitudinal 
            else self.ACTIONS_LAT
            if lateral 
            else None
        )
        
        if self.actions is None:
            raise ValueError("At least longitudinal or lateral actions must be included")
        
        self.target_speed = 0
        self.waypoint_idx = 0
        self.is_arrived = False

    def space(self) -> spaces.Discrete:
        return spaces.Discrete(len(self.actions))

        
    def setup_PID(self) -> PIDLongitudinalController:
        """
        Currently, the action is only supported for the longitudinal control.
        """

        args_lateral_dict = {
            'K_P': 1.95,
            'K_D': 0.2,
            'K_I': 0.07
            ,'dt': 1.0 / 10.0
        }

        args_long_dict = {
            'K_P': 1,
            'K_D': 0.0,
            'K_I': 0.75
            ,'dt': 1.0 / 10.0
        }

        self.controller = VehiclePIDController(self.controlled_vehicle, args_lateral=args_lateral_dict,args_longitudinal=args_long_dict)

    @property
    def controlled_vehicle(self):
        """
        The vehicle action upon.
        
        If not set, the first controlled vehicle is used by default.
        """
        return self._controlled_vehicle or self.env.vehicle
    
    @controlled_vehicle.setter
    def controlled_vehicle(self, vehicle):
        self._controlled_vehicle = vehicle
        self.setup_PID()

    @property
    def waypoints(self):
        return self._waypoints
    
    @waypoints.setter
    def waypoints(self, waypoints):
        self._waypoints = waypoints

    def calculate_distance(self, loc1: carla.Location, loc2: carla.Location) -> float:
        return loc1.distance(loc2)
    
    def act(self, action: np.ndarray) -> None:
        """
        Currently, the action is only supported for the longitudinal control.
        """

        self.env.world.debug.draw_point(self.waypoints[self.waypoint_idx].transform.location, size=0.1, color=carla.Color(0, 0, 255), life_time=0.1)

        action = self.actions[int(action)]
        
        if action == "FASTER":
            self.target_speed = min(80, self.target_speed + 1)
        elif action == "SLOWER":
            self.target_speed = max(0, self.target_speed - 1)
        elif action == "IDLE":
            pass

        if self.is_arrived:
            self.controlled_vehicle.apply_control(carla.VehicleControl(brake=1.0, steer=0.0))
        else:
            distance = self.calculate_distance(self.controlled_vehicle.get_location(), self.waypoints[self.waypoint_idx].transform.location)
            control = self.controller.run_step(self.target_speed, self.waypoints[self.waypoint_idx])
            self.controlled_vehicle.apply_control(control)

            if distance < 3.5:
                self.waypoint_idx += 1
            
            self._is_arrived()

    def _is_arrived(self) -> None:
        termination_point = carla.Transform(carla.Location(x=-16.789888, y=150.547333, z=0.281942), carla.Rotation(pitch=0, yaw=89.775124, roll=0))
        
        if self.calculate_distance(self.controlled_vehicle.get_location(), termination_point.location) < 3.5:
            self.is_arrived = True

class MultiAgentAction(ActionType):       
    def __init__(self, env: gym.Env, action_config: dict, **kwargs) -> None:
        super().__init__(env)
        if 'controlled_vehicles' in kwargs and 'waypoints' in kwargs:
            self.controlled_vehicles = kwargs['controlled_vehicles']
            self.waypoints = kwargs['waypoints']
            
            self.action_config = action_config
            self.agents_action_types = []

            for vehicle, waypoint in zip(self.controlled_vehicles, self.waypoints):
                action_type = action_factory(env, self.action_config)
                action_type.waypoints = waypoint
                action_type.controlled_vehicle = vehicle

                self.agents_action_types.append(action_type)

    def space(self) -> spaces.Space:
        return spaces.Tuple(
            [action_type.space() for action_type in self.agents_action_types]
        )
    
    def act(self, action: Action) -> None:
        assert isinstance(action, tuple)
        for agent_action, action_type in zip(action, self.agents_action_types):
            action_type.act(agent_action)

def action_factory(env: gym.Env, config: dict) -> ActionType:
    if config["type"] == "ContinuousAction":
        return ContinuousAction(env, **config)
    elif config["type"] == "DiscreteMetaAction":
        return DiscreteMetaAction(env, **config)
    elif config["type"] == "MultiAgentAction":
        return MultiAgentAction(env, **config)
    else:
        raise ValueError("Unknown action type {}".format(config["type"]))