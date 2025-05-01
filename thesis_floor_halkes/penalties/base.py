from abc import ABC, abstractmethod

from thesis_floor_halkes.action.base import Action
from thesis_floor_halkes.environment.base import Environment
# from thesis_floor_halkes.environment.dynamic_ambulance import AmbulanceEnvDynamic


class Penalty(ABC):
    """
    Abstract base class for penalties.
    """

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def __call__(self, environment:Environment, action:Action|None=None) -> float:
        """
        Call the penalty function.

        Args:
            environment: The environment to calculate the penalty for.
            action: The action taken in the environment.

        Returns:
            The calculated penalty.
        """
        pass
     