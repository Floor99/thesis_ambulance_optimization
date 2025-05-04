from thesis_floor_halkes.environment.base import Environment
from thesis_floor_halkes.penalties.base import Penalty


class PenaltyCalculator:
    
    def __init__(self, penalty_types: list[Penalty], penalty_weights: list[float]):
        """
        Initialize the penalty calculator with a list of penalties and their corresponding weights.
        
        Args:
            penalty_types (list[Penalty]): List of penalty types to be applied.
            penalty_weights (list[float]): List of weights for each penalty type.
        """
        self.penalty_types = penalty_types
        self.penalty_weights = penalty_weights
        assert len(penalty_types) == len(penalty_weights), "Penalty types and weights must have the same length."

    def calculate_penalty(self, environment:Environment, action) -> float:
        """
        Calculate the total penalty for the given environment.
        
        Args:
            environment (Environment): The environment for which to calculate the penalty.
        
        Returns:
            float: The total penalty.
        """
        total_penalty = 0.0
        for penalty_type, weight in zip(self.penalty_types, self.penalty_weights):
            total_penalty += penalty_type(environment, action) * weight
        return total_penalty
