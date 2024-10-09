from typing import Dict, List, Optional, Text, Tuple

import cv2
import numpy as np
import random

import carla
from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.tools.misc import get_speed

from carla_env.common.environment import CarlaEnv
from carla_env.common.action import Action
from carla_env.common.observation import Observation
from carla_env.common.utils import lmap

class Town04Env(CarlaEnv):
    """
    Environment class for reinforcement learning in CARLA simulation.
    using the Town04 map provided by CARLA, designed for multi-agent use.
    """

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                "action": {
                    "type": "DiscreteMetaAction",
                    "longitudinal": True,
                    "lateral": False
                },
                "observation": {
                    "type": "KinematicObservation",
                    "features": ["x", "y", "vx", "vy", "speed"],
                    "vehicles_count": 5,
                    "absolute": False,
                    "order": "sorted"
                },
                "duration": 5000,
                "collision_reward": -200,
                "high_speed_reward": 1,
                "reward_speed_range": [30, 80],
                "normalize_reward": False,
            }
        )

        return config

    def cleanup(self) -> None:
        for sensor in self.world.get_actors().filter('*sensor*'):
            sensor.destroy()
        for actor in self.world.get_actors().filter('*vehicle*'):
            actor.destroy()
        cv2.destroyAllWindows()

    def calculate_distance(self, loc1: carla.Location, loc2: carla.Location) -> float:
        return loc1.distance(loc2)
    
    def spawn_vehicles(self) -> None:
        """
        Spawn vehicles on the map.
        """
        num_CAV = 1
        num_HDV = random.randint(3, 5)

        self.cavs = []
        self.hdvs = []

        # Main road indexes
        self.main_road_indexes = [(1174, 6), (49, 6), (902, 6), (48, 6), (775, 6), (47, 6), (1073, 6), (46, 6)]

        # Merging road indexes
        self.merge_road_indexes = [(33, 2), (779, 2)]
        
        self.selected_spawn_points = []
        self.selected_end_points = []

        sampling_resolution = 0.2
        grp = GlobalRoutePlanner(self.world.get_map(), sampling_resolution)

        # Get spawn points on the main road
        main_road_start_point = carla.Transform(carla.Location(x=-13.909459, y=-252.608490, z=1.000000), carla.Rotation(pitch=360.000000, yaw=100.621307, roll=0.000000))
        main_road_end_point = carla.Transform(carla.Location(x=-16.789888, y=-170.547333, z=0.281942), carla.Rotation(pitch=0, yaw=89.775124, roll=0))

        main_road_spawn_points = []
        main_road_waypoints = grp.trace_route(main_road_start_point.location, main_road_end_point.location)

        for i in range(len(main_road_waypoints)):
            main_road_spawn_points.append(main_road_waypoints[i][0])

        # Get spawn points on the merging road
        merge_road_start_point = carla.Transform(carla.Location(x=-114.184532, y=-20.872791, z=7.083106), carla.Rotation(pitch=358.497162, yaw=242.975586, roll=0.000000))
        merge_road_end_point = carla.Transform(carla.Location(x=-83.570442, y=-95.477646, z=2.849217), carla.Rotation(pitch=356.027069, yaw=-20.431000, roll=0.000000))

        merge_road_spawn_points = []
        merge_road_waypoints = grp.trace_route(merge_road_start_point.location, merge_road_end_point.location)

        for i in range(len(merge_road_waypoints)):
            merge_road_spawn_points.append(merge_road_waypoints[i][0])

        # Maximum number of vehicles on each road
        num_main_road = 5
        num_merge_road = 5

        selected_main_spawn_points = []
        while len(selected_main_spawn_points) < num_main_road:
            candidate_point = random.choice(main_road_spawn_points)
            
            # Check distance to all selected main points
            if all(self.calculate_distance(candidate_point.transform.location, sp.transform.location) > 10 for sp in selected_main_spawn_points):
                selected_main_spawn_points.append(candidate_point)

        selected_merge_spawn_points = []
        while len(selected_merge_spawn_points) < num_merge_road:
            candidate_point = random.choice(merge_road_spawn_points)

            # Check distance to all selected merge points
            if all(self.calculate_distance(candidate_point.transform.location, sp.transform.location) > 10 for sp in selected_merge_spawn_points):
                selected_merge_spawn_points.append(candidate_point)

        end_point = carla.Transform(carla.Location(x=-108.876434, y=381.002808, z=0.001836), carla.Rotation(pitch=0.001393, yaw=151.651794, roll=-0.127411))

        vehicle_blueprints = [self.blueprint_library.find('vehicle.audi.etron'), self.blueprint_library.find('vehicle.dodge.charger_2020'), self.blueprint_library.find('vehicle.ford.mustang'), self.blueprint_library.find('vehicle.lincoln.mkz_2020'),
                              self.blueprint_library.find('vehicle.tesla.model3'), self.blueprint_library.find('vehicle.audi.a2'), self.blueprint_library.find('vehicle.audi.tt'), self.blueprint_library.find('vehicle.chevrolet.impala'),
                              self.blueprint_library.find('vehicle.dodge.charger_police'), self.blueprint_library.find('vehicle.ford.crown'), self.blueprint_library.find('vehicle.mini.cooper_s_2021')]
        
        # Randomly select spawn points for CAVs and HDVs
        if np.random.rand() < 0.5:
            spawn_point_cav_main = np.random.choice(selected_main_spawn_points, num_CAV, replace=False)
            spawn_point_cav = list(spawn_point_cav_main)
            selected_main_spawn_points.remove(spawn_point_cav[0])
        else:
            spawn_point_cav_merge = np.random.choice(selected_merge_spawn_points, num_CAV, replace=False)
            spawn_point_cav = list(spawn_point_cav_merge)
            selected_merge_spawn_points.remove(spawn_point_cav[0])

        spawn_point_hdv_main = np.random.choice(selected_main_spawn_points, num_HDV // 2, replace=False)
        spawn_point_hdv_merge = np.random.choice(selected_merge_spawn_points, num_HDV - num_HDV // 2, replace=False)

        spawn_point_hdv_main = list(spawn_point_hdv_main)
        spawn_point_hdv_merge = list(spawn_point_hdv_merge)

        # Spawn vehicles
        for waypoint in spawn_point_cav:
            spawn_point = waypoint.transform
            spawn_point.location.z += 1
            cav = self.world.spawn_actor(vehicle_blueprints[4], spawn_point)
            cav.apply_control(carla.VehicleControl(brake=1.0, steer=0.0))
            self.selected_spawn_points.append(spawn_point)
            self.cavs.append(cav)

        for waypoint in spawn_point_hdv_main:
            spawn_point = waypoint.transform
            spawn_point.location.z += 1
            random_index = random.choice(range(len(vehicle_blueprints)))
            hdv = self.world.spawn_actor(vehicle_blueprints[random_index], spawn_point)
            hdv.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0))
            self.hdvs.append(hdv)

        for waypoint in spawn_point_hdv_merge:
            spawn_point = waypoint.transform
            spawn_point.location.z += 1
            random_index = random.choice(range(len(vehicle_blueprints)))
            hdv = self.world.spawn_actor(vehicle_blueprints[random_index], spawn_point)
            hdv.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0))
            self.hdvs.append(hdv)

        # Set end points for all CAVs
        for i in range(num_CAV):
            self.selected_end_points.append(end_point)
            
        self.vehicles = self.cavs + self.hdvs

    def get_initial_waypoints(self) -> List[Tuple[carla.Waypoint, carla.Waypoint]]:
        waypoints = []

        sampling_resolution = 0.2
        
        grp = GlobalRoutePlanner(self.world.get_map(), sampling_resolution)

        # Get waypoints on the route using selected_spawn_points and selected_end_points
        for selected_spawn_point, selected_end_point in zip(self.selected_spawn_points, self.selected_end_points):
            route = grp.trace_route(selected_spawn_point.location, selected_end_point.location)
            route = [route[i][0] for i in range(len(route))]

            waypoints.append(route)

        return waypoints
    
    def reset(self) -> Tuple[Observation, dict]:
        self.steps = 0
        self.done = False

        self.local_reward = dict()
        self.collision_dict = dict()

        self._reset()
        self.world.tick()

        for vehicle in self.cavs:
            self.local_reward[vehicle] = 0
            self.collision_dict[vehicle] = False

        obs = self.observation_type.observe()

        info = self._info(obs, action=self.action_space.sample())

        obs = np.asarray(obs).reshape((len(obs), -1))

        return obs, info
    
    def _reset(self) -> None:
        self.cleanup()
        self.spawn_vehicles()

        self.define_spaces()

        self.action_type.waypoints = self.get_initial_waypoints()[0]
        self.action_type.controlled_vehicle = self.cavs[0]

        self.observation_type.observer_vehicle = self.cavs[0]

        self.observation_type.vehicles = self.vehicles

        self.collision_hist = []
        self.actor_list = []

        for cav in self.cavs:
            sensor_init_transform = carla.Transform(carla.Location(z=1.3,x=1.4))
            cav.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))

            colsensor = self.blueprint_library.find("sensor.other.collision")
            self.colsensor = self.world.spawn_actor(colsensor, sensor_init_transform, attach_to=cav)
            self.colsensor.listen(lambda event: self.collision_data(event, cav))
            self.actor_list.append(self.colsensor)

            cav.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))

        if self.hdvs:
            for hdv in self.hdvs:
                hdv.set_autopilot(True)

    def collision_data(self, event, vehicle) -> None:
        self.collision_dict[vehicle] = True
        self.collision_hist.append(event)

    def step(self, action: Action) -> Tuple[Observation, float, bool, dict]:
        """
        Perform an action and step the environment dynamics.

        The action is executed by the ego-vehicle, and all other vehicles on the road performs their default behaviour
        
        :param action: the action performed by the ego-vehicle
        :return: a tuple (observation, reward, terminated, truncated, info)
        """

        self.steps += 1
        
        self._simulate(action)

        obs = self.observation_type.observe()
        obs = np.asarray(obs).reshape((len(obs), -1))
        reward = self._reward(action)

        truncated = self._is_truncated()
        terminated = self._is_terminated()
        info = self._info(obs, action)

        return obs, reward, terminated, truncated, info
    
    def _reward(self, action: int) -> float:
        # Initialize a variable to hold the sum of rewards
        total_reward = 0

        self.local_reward[self.cavs[0]] = self._agent_reward(self.cavs[0])
        total_reward += self.local_reward[self.cavs[0]]

        return total_reward / len(self.cavs)
    
    def _agent_reward(self, vehicle: carla.Vehicle) -> float:
        """Per-agent reward signal."""
        scaled_speed = lmap(get_speed(vehicle), self.config["reward_speed_range"], [0, 1])    

        reward = self.config["collision_reward"] * self.collision_dict[vehicle] \
                + self.config["high_speed_reward"] * scaled_speed
        
        return reward
    
    def _simulate(self, action: Optional[Action]):
        self.action_type.act(action)
        self.world.tick()

    def _is_terminated(self) -> bool:
        # If the vehicle collides or arrives, the episode ends
        # TODO: 모든 차량이 도착 했을 때 충돌이 발생하지 않는지 확인 필요.
        return self.action_type.is_arrived or self.collision_dict[self.cavs[0]]
    
    def _is_truncated(self) -> bool:
        # If the duration exceeds, the episode is terminated early
        return self.steps >= self.config["duration"]