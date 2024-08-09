from typing import Dict, Optional, Text, Tuple

import gym

import carla

from common.observation import Observation, ObservationType, observation_factory
from common.action import Action, ActionType, action_factory

class CarlaEnv(gym.Env):
    action_type: ActionType
    observation_type: ObservationType

    def __init__(self, config: Dict = None) -> None:
        super().__init__()

        # CARLA Simulation
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(4.0)
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()

        # Configuration
        self.config =self.default_config()
        self.configure(config)

        # Spaces
        self.action_type = None
        self.action_space = None
        self.observation_type = None
        self.observation_space = None

        # Running
        self.time = 0  # Simulation time
        self.steps = 0  # Actions performed
        self.done = False  # Episode termination

        self.reset()
        
    @classmethod
    def default_config(cls) -> Dict:
        """
        Default environment configuration.

        Can be overloaded in environment implementations, or by calling configure().
        :return: a configuration dict
        """
        return {
            "observation": {
                "type": "Kinematics",
                "features": ["x", "y", "z", "picth", "yaw", "roll", "vx", "vy", "vz"],
                "vehicles_count": 5,
            },
            "action": {
                "type": "ParameterTuningAction",
                "Kp_range": (0.0, 10.0),
                "Ki_range": (0.0, 10.0),
                "Kd_range": (0.0, 10.0),
                "desired_speed_range": (0.0, 10.0),
                "look_ahead_distance_range": (20.0, 50.0)
            }
        }
    
    def configure(self, config: Dict) -> None:
        """
        Overload configuration with a new one.

        :param config: a configuration dict
        """
        if config:
            self.config.update(config)

    def define_spaces(self) -> None:
        """
        Set the types and spaces of observation and action from config.
        """
        self.observation_type = observation_factory(self.world, self.config["observation"])
        self.observation_space = self.observation_type.space()
        
        self.action_type = action_factory(self.world, self.config["action"])
        self.action_space = self.action_type.space()
    
    def _reward(self, action: Action) -> float:
        """
        Return the reward associated with performing a given action and ending up in the current state.

        :param action: the last action performed
        :return: the reward
        """
        raise NotImplementedError

    def _rewards(self, action: Action) -> Dict[Text, float]:
        """
        Returns a multi-objective vector of rewards.

        If implemented, this reward vector should be aggregated into a scalar in _reward().
        This vector value should only be returned inside the info dict.

        :param action: the last action performed
        :return: a dict of {'reward_name': reward_value}
        """
        raise NotImplementedError
    
    def _is_terminated(self) -> bool:
        """
        Check whether the current state is a terminal state

        :return:is the state terminal
        """
        raise NotImplementedError
    
    def _is_truncated(self) -> bool:
        """
        Check we truncate the episode at the current step

        :return: is the episode truncated
        """
        raise NotImplementedError
    
    def _info(self, obs: Observation, action: Optional[Action] = None) -> dict:
        """
        Return a dictionary of additional information

        :param obs: current observation
        :param action: current action
        :return: info dict
        """
        info = {
            "action": action,
        }
        try:
            info["rewards"] = self._rewards(action)
        except NotImplementedError:
            pass
        return info
    
    def reset(self) -> Tuple[Observation, dict]:
        """
        Reset the environment to it's initial configuration

        :return: the observation of the reset state
        """
        
        self.time = self.steps = 0
        self.done = False

        self._reset()

        obs = self.observation_type.observe()
        # TODO 'info' is not used yet
        info = self._info(obs, action=self.action_space.sample())

        return obs

    def _reset(self) -> None:
        """
        Reset the scene: remove the previous actors and spawn new ones.

        This method must be overloaded by the environments.
        """
        raise NotImplementedError()
    
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
        reward = self._reward(action)
        terminated = self._is_terminated()
        # TODO 'truncated' and 'info' are not used yet
        truncated = self._is_truncated()
        info = self._info(obs, action)

        return obs, reward, terminated, {}
    
    def _simulate(self, action: Optional[Action] = None) -> None:
        """The ego-vehicle performs the action."""
        raise NotImplementedError()