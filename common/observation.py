from typing import Dict, List, TypeVar

import numpy as np
import pandas as pd

from gym import spaces

import carla

Observation = TypeVar("Observation")

class ObservationType(object):
    def __init__(self, world, **kwargs: Dict) -> None:
        self.world = world
        self.vehicles = None
        self.__observer_vehicle = None

    def space(self) -> spaces.Space:
        """ Get the observation space. """
        raise NotImplementedError
    
    def observe(self) -> object:
        """ Get an observation of the environment state. """
        raise NotImplementedError
    
    @property
    def vehicles(self):
        return self.__vehicles
    
    @vehicles.setter
    def vehicles(self, vehicles):
        self.__vehicles = vehicles

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
    def __init__(self,
                 world: carla.World,
                 features: List[str] = None,
                 vehicles: List[carla.Vehicle] = None,
                 vehicles_count: int = 5,
                 order: str = "sorted",
                 display: bool = True,
                 **kwargs: Dict) -> None:
        """
        Observe the kinematics of nearby vehicles.
        :param world: the CARLA Simulation
        :param features: the features to include in the observation
        :vehicles: the list of all vehicles in simulation
        :vehicles_count: the number of vehicles to include in the observation
        :order: the order in which to include vehicles in the observation (sorted or shuffled)
        :display: whether to display the observation in the simulation
        """
        super().__init__(world)
        self.features = features or ["presence", "x", "y", "z", "picth", "yaw", "roll", "vx", "vy", "vz"]
        self.vehicles = vehicles
        self.vehicles_count = vehicles_count
        self.order = order
        self.display = display

    def space(self) -> spaces.Box:
        return spaces.Box(
            shape=(self.vehicles_count, len(self.features)),
            low=-np.inf,
            high=np.inf,
            dtype=np.float32
        )

    def vehicle_to_dict(self, vehicle: carla.Vehicle) -> Dict:
        """Attempts to get the transform of a vehicle, returning None if the vehicle is destroyed."""
        try:
            transform = vehicle.get_transform()
        except RuntimeError as e:
            print(f"Error getting transform for vehicle {vehicle}: {e}")
            return None
            
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
            "vz": round(velocity.z, 3)
        }
        return d
    
    def vector3d_to_ndarray(self, vector3d: carla.Vector3D) -> np.ndarray:
        return np.array([vector3d.x, vector3d.y, vector3d.z])
    
    def get_location(self, vehicle: carla.Vehicle) -> np.ndarray:
        """Attempts to get the location of a vehicle, returning None if the vehicle is destroyed."""
        try:
            return self.vector3d_to_ndarray(vehicle.get_location())
        except RuntimeError as e:
            print(f"Error getting location for vehicle {vehicle}: {e}")
            return None
    
    def get_distance_between_vehicles(self, ego_vehicle: carla.Vehicle, other_vehicle: carla.Vehicle) -> float:
        if ego_vehicle is not other_vehicle:
            ego_location = self.get_location(ego_vehicle)
            other_location = self.get_location(other_vehicle)
            
            if ego_location is not None and other_location is not None:
                return np.linalg.norm(other_location - ego_location)
            else:
                # If any of the vehicles is destroyed, return NaN
                return np.nan
        else:  # same vehicle
            return np.nan
        
    def sort_vehicles_by_distance(self, ego_vehicle: carla.Vehicle, vehicles: List[carla.Vehicle]) -> List[carla.Vehicle]:
        vehicles_without_ego = [v for v in vehicles if v is not ego_vehicle]
        sorted_vehicles = sorted(vehicles_without_ego, key=lambda v: self.get_distance_between_vehicles(ego_vehicle, v))
        return sorted_vehicles
    
    def observe(self) -> np.ndarray:
        if not self.world:
            return np.zeros(self.space().shape)
        
        df = pd.DataFrame.from_records([self.vehicle_to_dict(self.observer_vehicle)])
    
        vehicles_df = pd.DataFrame.from_records(
            [
                self.vehicle_to_dict(v)
                for v in self.sort_vehicles_by_distance(self.observer_vehicle, self.vehicles)
            ]
        )
        
        if self.order == "shuffled":
            vehicles_df = vehicles_df.sample(frac=1).reset_index(drop=True)

        df = pd.concat([df, vehicles_df], ignore_index=True)
        
        # Cut dataFrame rows if it has more rows than vehicles_count
        if df.shape[0] > self.vehicles_count:
            df = df.iloc[:self.vehicles_count]

        # Fill missing rows
        if df.shape[0] < self.vehicles_count:
            rows = np.zeros((self.vehicles_count - df.shape[0], len(self.features)))
            df = pd.concat(
                [df, pd.DataFrame(data=rows, columns=self.features)], ignore_index=True
            )

        # Remove the columns that are not in the features list
        df = df[self.features]
        obs = df.values.copy()

        # Display the location of the observed vehicles in the simulation (ego vehicle in yellow, others in red)
        if self.display:
            self.display_observation(df)
        
        # Flatten
        return obs.astype(self.space().dtype)
    
    def display_observation(self, df: pd.DataFrame) -> None:
        for idx, row in df.iterrows():
            location = carla.Location(x=row["x"], y=row["y"], z=row["z"])
            
            # If index is 0 then color is yellow, otherwise red
            color = carla.Color(255, 255, 0) if idx == 0 else carla.Color(255, 0, 0)
            
            self.world.debug.draw_point(location, size=0.2, color=color, life_time=0.2)

class MultiAgentObservation(ObservationType):   
    def __init__(self, world: carla.World, observation_config: Dict, **kwargs: Dict) -> None:
        super().__init__(world)
        if 'observer_vehicles' in kwargs and 'vehicles' in kwargs:
            self.observer_vehicles = kwargs['observer_vehicles']
            self.vehicles = kwargs['vehicles']

            self.observation_config = observation_config
            self.agents_observation_types = []
            for vehicle in self.observer_vehicles:
                observation_type = observation_factory(world, self.observation_config)
                observation_type.observer_vehicle = vehicle
                observation_type.vehicles = self.vehicles
                self.agents_observation_types.append(observation_type)

    def space(self) -> spaces.Tuple:
        return spaces.Tuple([obs_type.space() for obs_type in self.agents_observation_types])
    
    def observe(self) -> tuple:
        return tuple(obs_type.observe() for obs_type in self.agents_observation_types)
    
def observation_factory(world: carla.World, config: Dict) -> ObservationType:
    if config["type"] == "KinematicObservation":
        return KinematicObservation(world, **config)
    elif config["type"] == "MultiAgentObservation":
        return MultiAgentObservation(world, **config)
    else:
        raise ValueError("Unknown action type {}".format(config["type"]))