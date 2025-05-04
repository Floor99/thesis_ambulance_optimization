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
    def __call__(self, environment:Environment, penalty:float, action:Action|None=None) -> float:
        """
        Call the penalty function.

        Args:
            environment: The environment to calculate the penalty for.
            action: The action taken in the environment.

        Returns:
            The calculated penalty.
        """
        pass

class Bonus(ABC):
    """
    Abstract base class for bonuses.
    """

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def __call__(self, environment:Environment, bonus:float, action:Action|None=None) -> float:
        """
        Call the bonus function.

        Args:
            environment: The environment to calculate the bonus for.
            action: The action taken in the environment.

        Returns:
            The calculated bonus.
        """
        pass     