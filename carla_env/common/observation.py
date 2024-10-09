from typing import Dict, List, TypeVar

import numpy as np
import pandas as pd

import gym
from gym import spaces

from agents.tools.misc import get_speed

import carla

Observation = TypeVar("Observation")

class ObservationType:
    def __init__(self, env: gym.Env, **kwargs) -> None:
        self.env = env
        self.__observer_vehicle = None

    def space(self) -> spaces.Space:
        """Get the observation space."""
        raise NotImplementedError
    
    def observe(self) -> None:
        """Get an observation of the environment state."""
        raise NotImplementedError()
    
    @property
    def observer_vehicle(self):
        """
        The vehicle observing the scene.

        If not set, the first controlled vehicle is used by default.
        """
        return self.__observer_vehicle

    @observer_vehicle.setter
    def observer_vehicle(self, vehicle):
        self.__observer_vehicle = vehicle

class KinematicObservation(ObservationType):
    """Observe the kinematics of nearby vehicles."""

    def __init__(
            self,
            env: gym.Env,
            features: List[str] = None,
            vehicles_count: int = 5,
            order: str = "sorted",
            **kwargs: Dict
    ) -> None:
        """
        :param env: The environment to observe
        :param features: Names of features used in the observation
        :param vehicles_count: Number of observed vehicles
        :param order: Order of observed vehicles. Values: sorted, shuffled
        """

        super().__init__(env)
        self.features = features or self.FEATURES
        self.vehicles_count = vehicles_count
        self.order = order

    def space(self) -> spaces.Box:
        return spaces.Box(
            shape=(self.vehicles_count, len(self.features)),
            low=-np.inf,
            high=np.inf,
            dtype=np.float32
        )
    
    def vehicle_to_dict(self, vehicle: carla.Vehicle) -> dict:
        """Attempts to get the transform of a vehicle, returning None if the vehicle is destroyed."""

        transform = vehicle.get_transform()
        location = transform.location
        rotation = transform.rotation
        
        velocity = vehicle.get_velocity()

        d = {
            "presence": 1,
            "x": round(location.x, 3),
            "y": round(location.y, 3),
            "z": round(location.z, 3),
            "picth": round(rotation.pitch, 3),
            "yaw": round(rotation.yaw, 3),
            "roll": round(rotation.roll, 3),
            "vx": round(velocity.x, 3),
            "vy": round(velocity.y, 3),
            "vz": round(velocity.z, 3),
            "speed": round(get_speed(vehicle), 3),
        }
        return d
    
    def calculate_distance(self, loc1: carla.Location, loc2: carla.Location) -> float:
        return loc1.distance(loc2)

    def sort_vehicles_by_distance(self, ego_vehicle: carla.Vehicle, vehicles: List[carla.Vehicle]) -> List[carla.Vehicle]:
        vehicles_without_ego = [v for v in vehicles if v is not ego_vehicle]
        sorted_vehicles = sorted(vehicles_without_ego, key=lambda v: self.calculate_distance(ego_vehicle.get_location(), v.get_location()))
        return sorted_vehicles
    
    def observe(self) -> np.ndarray:
        if not self.env:
            return np.zeros(self.space().shape)
        
        dp = pd.DataFrame.from_records([self.vehicle_to_dict(self.observer_vehicle)])

        vehicles_df = pd.DataFrame.from_records(
            [
                self.vehicle_to_dict(v)
                for v in self.sort_vehicles_by_distance(self.observer_vehicle, self.env.vehicles)
            ]
        )

        if self.order == "shuffled":
            vehicles_df = vehicles_df.sample(frac=1).reset_index(drop=True)

        df = pd.concat([dp, vehicles_df], ignore_index=True)

        # Cut dataframe rows to match the vehicles count
        if df.shape[0] > self.vehicles_count:
            df = df.iloc[:self.vehicles_count]

        # Fill missing rows with zeros
        if df.shape[0] < self.vehicles_count:
            rows = np.zeros((self.vehicles_count - df.shape[0], len(self.features)))
            df = pd.concat([df, pd.DataFrame(data=rows, columns=self.features)], ignore_index=True)
        
        # Remove the columns that are not in the features list
        df = df[self.features]
        obs = df.values.copy()

        return obs.astype(self.space().dtype)
    
class MultiAgentObservation(ObservationType):   
    def __init__(self, world: carla.World, observation_config: dict, **kwargs: dict) -> None:
        super().__init__(world)
        if 'observer_vehicles' in kwargs:
            self.observer_vehicles = kwargs['observer_vehicles']

            self.observation_config = observation_config
            self.agents_observation_types = []
            for vehicle in self.observer_vehicles:
                observation_type = observation_factory(world, self.observation_config)
                observation_type.observer_vehicle = vehicle
                self.agents_observation_types.append(observation_type)

    def space(self) -> spaces.Tuple:
        return spaces.Tuple([obs_type.space() for obs_type in self.agents_observation_types])
    
    def observe(self) -> tuple:
        return tuple(obs_type.observe() for obs_type in self.agents_observation_types)
    
def observation_factory(env: gym.Env, config: dict) -> ObservationType:
    if config["type"] == "KinematicObservation":
        return KinematicObservation(env, **config)
    elif config["type"] == "MultiAgentObservation":
        return MultiAgentObservation(env, **config)
    else:
        raise ValueError("Unknown observation type {}".format(config["type"]))