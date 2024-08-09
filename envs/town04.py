import sys
sys.path.append('C:/Users/seominseok/carla/PythonAPI/carla')

from typing import Dict, List, Optional, Text, Tuple

import cv2
import time
import random
import numpy as np
from math import *

import carla
from agents.navigation.global_route_planner import GlobalRoutePlanner

from common.environment import CarlaEnv
from common.action import Action
from common.observation import Observation

import common.utils as utils
from controller import Controller

class Town04Env(CarlaEnv):
    """
    Environment class for reinforcement learning in CARLA simulation.
    using the Town04 map provided by CARLA, designed for multi-agent use.

    :param action: the action space of the environment
    :param observation: the observation space of the environment
    :param duration: the duration of the simulation in steps
    :param collision_reward: the reward for a collision
    :param high_speed_reward: the reward for high speed
    :param arrived_reward: the reward for arriving at the destination
    :param reward_speed_range: the range of speeds for the reward function
    :param normalize_reward: whether to normalize the reward function
    """
    @classmethod
    def default_config(cls) -> Dict:
        config = super().default_config()
        config.update(
            {   
                "action": {
                    "type": "ParameterTuningAction",
                    "Kp_range": (0.0, 30.0),
                    "Ki_range": (0.0, 30.0),
                    "Kd_range": (0.0, 30.0),
                    "desired_speed_range": (10.0, 30.0),
                    "look_ahead_distance_range": (20.0, 50.0)
                },
                "observation": {
                    "type": "KinematicObservation",
                    "features": ["x", "y", "z", "picth", "yaw", "roll", "vx", "vy", "vz"],
                    "vehicles_count": 5
                },
                "duration": 1500,
                "collision_reward": -5,
                "high_speed_reward": 1,
                "arrived_reward": 1,
                "reward_speed_range": [0, 30],
                "normalize_reward": False,
            }
        )

        return config
    
    def cleanup(self) -> None:
        if hasattr(self, 'actor_list') and self.actor_list:
            for actor in self.actor_list:
                actor.destroy()

        for sensor in self.world.get_actors().filter('*sensor*'):
            sensor.destroy()
        for actor in self.world.get_actors().filter('*vehicle*'):
            actor.destroy()

        cv2.destroyAllWindows()

    def spawn_vehicles(self) -> None:
        self.cavs = []
        self.hdvs = []
        
        self.selected_spawn_points = []
        self.selected_end_points = []
        
        """
        Spawn Points and End Points:

        - The first line represents a 3-lane road.
        - The second line represents a 4-lane road.
        - The third line represents a merging road.
        """

        spawn_points = [carla.Transform(carla.Location(x=-13.299323, y=-167.961075, z=0.281942), carla.Rotation(pitch=0, yaw=89.775124, roll=0)), carla.Transform(carla.Location(x=-13.299323, y=-195.961075, z=0.281942), carla.Rotation(pitch=0, yaw=89.775124, roll=0)), carla.Transform(carla.Location(x=-13.299323, y=-220.547333, z=0.281942), carla.Rotation(pitch=0, yaw=89.775124, roll=0)),
                        carla.Transform(carla.Location(x=-16.789888, y=-160.547333, z=0.281942), carla.Rotation(pitch=0, yaw=89.775124, roll=0)), carla.Transform(carla.Location(x=-16.789888, y=-185.547333, z=0.281942), carla.Rotation(pitch=0, yaw=89.775124, roll=0)), carla.Transform(carla.Location(x=-16.789888, y=-200.547333, z=0.281942), carla.Rotation(pitch=0, yaw=89.775124, roll=0)), carla.Transform(carla.Location(x=-16.789888, y=-230.547333, z=0.281942), carla.Rotation(pitch=0, yaw=89.775124, roll=0)),
                        carla.Transform(carla.Location(x=-119.409157, y=-39.837208, z=9.003113), carla.Rotation(pitch=0, yaw=-90.775124, roll=0)), carla.Transform(carla.Location(x=-118.909157, y=-50.837208, z=8.003113), carla.Rotation(pitch=0, yaw=-80.775124, roll=0)), carla.Transform(carla.Location(x=-110.909157, y=-73.437208, z=6.003113), carla.Rotation(pitch=0, yaw=-58.775124, roll=0)), carla.Transform(carla.Location(x=-100.909157, y=-84.837208, z=5.003113), carla.Rotation(pitch=0, yaw=-45.775124, roll=0)), carla.Transform(carla.Location(x=-90.909157, y=-91.837208, z=4.003113), carla.Rotation(pitch=0, yaw=-27.775124, roll=0)), carla.Transform(carla.Location(x=-70.909157, y=-98.837208, z=2.003113), carla.Rotation(pitch=0, yaw=-5.775124, roll=0))]

        end_points = [carla.Transform(carla.Location(x=-11.527476, y=210.010345, z=0.281942), carla.Rotation(pitch=0.001004, yaw=89.691406, roll=0.000371)),
                      carla.Transform(carla.Location(x=-15.115714, y=210.010345, z=0.281942), carla.Rotation(pitch=-0.002288, yaw=89.718376, roll=-0.000214)),
                      carla.Transform(carla.Location(x=-124.204468, y=80.110641, z=8.163152), carla.Rotation(pitch=4.620616, yaw=-79.688652, roll=-0.124115))]
                
        vehicle_blueprints = [self.blueprint_library.find('vehicle.audi.etron'), self.blueprint_library.find('vehicle.dodge.charger_2020'), self.blueprint_library.find('vehicle.ford.mustang'), self.blueprint_library.find('vehicle.lincoln.mkz_2020'),
                              self.blueprint_library.find('vehicle.tesla.model3'), self.blueprint_library.find('vehicle.audi.a2'), self.blueprint_library.find('vehicle.audi.tt'), self.blueprint_library.find('vehicle.chevrolet.impala'),
                              self.blueprint_library.find('vehicle.dodge.charger_police'), self.blueprint_library.find('vehicle.ford.crown'), self.blueprint_library.find('vehicle.mini.cooper_s_2021')]
        

        self.vehicles_count = len(spawn_points)

        indexes = list(range(len(spawn_points)))
        cav_index = random.choice(indexes)
        end_index = random.choice(range(len(end_points)))
        hdv_indexes = list(set(indexes) - set([cav_index]))

        cav = self.world.spawn_actor(vehicle_blueprints[4], spawn_points[cav_index])
        self.cavs.append(cav)

        self.selected_spawn_point = spawn_points[cav_index]
        self.selected_end_point = end_points[end_index]

        for idx in hdv_indexes:
            random_index = random.choice(range(len(vehicle_blueprints)))
            hdv = self.world.spawn_actor(vehicle_blueprints[random_index], spawn_points[idx])
            self.hdvs.append(hdv)
            
        self.vehicles = self.cavs + self.hdvs

    def get_vehicle_waypoints(self) -> List[Tuple[carla.Waypoint, carla.Waypoint]]:
        waypoints_list = []

        sampling_resolution = 1
        
        grp = GlobalRoutePlanner(self.world.get_map(), sampling_resolution)

        # Get the waypoints from the spawn point to the end point
        route = grp.trace_route(self.selected_spawn_point.location, self.selected_end_point.location)

        waypoints = []
        debug_waypoints = []

        for waypoint in route:
            temp1 = []
            temp1.append(waypoint[0].transform.location.x)
            temp1.append(waypoint[0].transform.location.y)
            temp1.append(8.0)
            waypoints.append(temp1)

            temp2 = []
            temp2.append(waypoint[0].transform.location.x)
            temp2.append(waypoint[0].transform.location.y)
            temp2.append(waypoint[0].transform.location.z)
            debug_waypoints.append(temp2)

            self.world.debug.draw_string(waypoint[0].transform.location, '^', draw_shadow=False, color=carla.Color(r=0, g=0, b=255), life_time=20.0, persistent_lines=True)

        waypoints = np.array(waypoints)
        debug_waypoints = np.array(debug_waypoints)

        waypoints_list.append((waypoints, debug_waypoints))

        return waypoints_list
    
    def step(self, action: Action) -> Tuple[Observation, float, bool, bool, dict]:
        """
        Perform an action and step the environment dynamics.

        The action is executed by the ego-vehicle, and all other vehicles on the road performs their default behaviour
        for several simulation timesteps until the next decision making step.

        :param action: the action performed by the ego-vehicle
        :return: a tuple (observation, reward, terminated, truncated, info)
        """
        self.steps += 1
        self._simulate(action)

        obs = self.observation_type.observe()
        reward = self._reward(action)
        terminated = self._is_terminated()
        truncated = self._is_truncated()
        # TODO 'info' is not used yet
        info = self._info(obs, action)

        # If ego vehicle satisfies the termination condition or the time is up, the episode ends
        if terminated or truncated:
            terminated = True

        return obs, reward, terminated, {}
    
    def _reset(self) -> None:
        # Remove the previous actors and spawn new ones
        self.cleanup()
        self.spawn_vehicles()

        # Define action and observation spaces
        self.define_spaces()

        waypoints_list = self.get_vehicle_waypoints()
        waypoints, debug_waypoints = waypoints_list[0]

        self.action_type.waypoints = waypoints
        self.action_type.debug_waypoints = debug_waypoints
        self.action_type.controlled_vehicle = self.cavs[0]
        self.action_type.controller = Controller(self.action_type.waypoints)

        self.observation_type.observer_vehicle = self.cavs[0]
        self.observation_type.vehicles = self.vehicles
        
        self.collision_hist = []
        self.lane_invade_hist = []
        self.actor_list = []
        
        sensor_init_transform = carla.Transform(carla.Location(z=1.3,x=1.4))
        self.action_type.controlled_vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
        time.sleep(2)

        colsensor = self.blueprint_library.find("sensor.other.collision")
        self.colsensor = self.world.spawn_actor(colsensor, sensor_init_transform, attach_to=self.action_type.controlled_vehicle)
        self.colsensor.listen(lambda event: self.collision_data(event))
        self.actor_list.append(self.colsensor)

        lanesensor = self.blueprint_library.find("sensor.other.lane_invasion")
        self.lanesensor = self.world.spawn_actor(lanesensor, sensor_init_transform, attach_to=self.action_type.controlled_vehicle)
        self.lanesensor.listen(lambda event: self.lane_data(event))
        self.actor_list.append(self.lanesensor)

        self.action_type.controlled_vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
        
        if self.hdvs:
            for hdv in self.hdvs:
                hdv.set_autopilot(True)

    def collision_data(self, event):
        self.collision_hist.append(event)
        
    def lane_data(self, event):
        self.lane_invade_hist.append(event)

    # Call from _info method
    def _rewards(self, action: int) -> Dict[Text, float]:
        """Multi-objective rewards, for cooperative agents."""
        agents_rewards = [
            self._agent_rewards(action, vehicle) for vehicle in self.cavs
        ]

        return {
            name: sum(agent_rewards[name] for agent_rewards in agents_rewards)
            / len(agents_rewards)
            for name in agents_rewards[0].keys()
        }
    
    """
    Method Call Sequence: _reward method -> _agent_reward method -> _agent_rewards method
    """
    def _reward(self, action: int) -> float:
        """Aggregated reward, for cooperative agents."""
        # Sum the rewards of each agent and divide by the number of agents
        return sum(
            self._agent_reward(action, vehicle) for vehicle in self.cavs
        ) / len(self.cavs)

    def _agent_reward(self, action: int, vehicle: carla.Vehicle) -> float:
        """Per-agent reward signal."""
        rewards = self._agent_rewards(action, vehicle)

        reward = sum(
            self.config.get(name, 0) * reward for name, reward in rewards.items()
        )

        # If the vehicle has arrived, only the arrived reward is received
        reward = self.config["arrived_reward"] if rewards["arrived_reward"] else reward
        
        # If vehicle is not on the road, the reward is 0
        reward *= rewards["on_road_reward"]

        if self.config["normalize_reward"]:
            reward = utils.lmap(
                reward,
                [self.config["collision_reward"], self.config["arrived_reward"]],
                [0, 1],
            )
        return reward

    def _agent_rewards(self, action: int, vehicle: carla.Vehicle) -> Dict[Text, float]:
        """Per-agent per-objective reward signal."""

        scaled_speed = utils.lmap(
            self.action_type.get_current_velocity(self.action_type.controlled_vehicle), self.config["reward_speed_range"], [0, 1]
        )

        return {
            "collision_reward": len(self.collision_hist) != 0,
            "high_speed_reward": np.clip(scaled_speed, 0, 1),
            "arrived_reward": self.has_arrived(self.action_type.controlled_vehicle),
            "on_road_reward": len(self.lane_invade_hist) == 0,
        }
    
    def _simulate(self, action: Optional[Action]) -> bool:
        # If the vehicle fails to operate and loses waypoints, the episode ends
        if self.action_type.act(action):
            return True
        
        return False
    
    def _is_terminated(self) -> bool:
        # If the vehicle collides or arrives, the episode ends
        return len(self.collision_hist) != 0 or self.has_arrived(self.action_type.controlled_vehicle)
    
    def _is_truncated(self) -> bool:
        # If the duration exceeds, the episode is terminated early
        return self.steps >= self.config["duration"]
    
    def calculate_distance(self, x1: float, y1: float, x2: float, y2: float) -> float:
        return sqrt((x2 - x1) ** 2 + (y2 - y1)  ** 2)
    
    def has_arrived(self, vehicle: carla.Vehicle) -> bool:
        """
        Check if the vehicle has arrived at the destination.
        """
        terminate_points = [carla.Transform(carla.Location(x=-13.299323, y=150.961075, z=0.281942), carla.Rotation(pitch=0, yaw=89.775124, roll=0)),
                            carla.Transform(carla.Location(x=-16.789888, y=150.547333, z=0.281942), carla.Rotation(pitch=0, yaw=89.775124, roll=0)),
                            carla.Transform(carla.Location(x=-94.992263, y=139.336074, z=1.6782921), carla.Rotation(pitch=3.012044, yaw=-154.70425, roll=-0.539733))]
        
        self.world.tick()
        vehicle_location = vehicle.get_location()

        for terminate_point in terminate_points:
            terminate_location = terminate_point.location
            if self.calculate_distance(vehicle_location.x, vehicle_location.y, terminate_location.x, terminate_location.y) < 10:
                return True
        
        return False