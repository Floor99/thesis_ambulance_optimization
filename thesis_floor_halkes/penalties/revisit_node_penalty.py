from typing import List
from thesis_floor_halkes.environment.base import Environment
from thesis_floor_halkes.penalties.base import Penalty, Bonus


class RevisitNodePenalty(Penalty):
    """
    Penalty for revisiting a node in the environment.
    """

    def __init__(self, name: str, penalty:float):
        super().__init__(name, penalty)
        
    def __call__(self, **kwargs) -> float:
        """
        Calculate the revisit node penalty.

        Args:
            environment: The environment to calculate the penalty for.
            action: The action taken in the environment.

        Returns:
            The calculated penalty.
        """
        visited_nodes = kwargs.get('visited_nodes', List[int])
        action = kwargs.get('action', int)
        
        if action in visited_nodes:
            return self.penalty
        return 0.0

class PenaltyPerStep(Penalty):
    """
    Penalty for each step taken in the environment.
    """

    def __init__(self, name: str, penalty:float):
        super().__init__(name, penalty)

    def __call__(self, **kwargs) -> float:
        """
        Calculate the penalty per step.

        Args:
            environment: The environment to calculate the penalty for.
            action: The action taken in the environment.

        Returns:
            The calculated penalty.
        """
        return self.penalty

class GoalBonus(Bonus):
    """
    Bonus for reaching the goal in the environment.
    """

    def __init__(self, name: str, bonus:float):
        super().__init__(name, bonus)

    def __call__(self, **kwargs) -> float:
        """
        Calculate the goal bonus.

        Args:
            environment: The environment to calculate the bonus for.
            action: The action taken in the environment.

        Returns:
            The calculated bonus.
        """
        current_node = kwargs.get('current_node', int)
        end_node = kwargs.get('end_node', int)
        
        if current_node == end_node:
            return self.bonus
        return 0.0

class DeadEndPenalty(Penalty):
    """
    Penalty for reaching a dead end in the environment.
    """

    def __init__(self, name: str, penalty:float):
        super().__init__(name, penalty)

    def __call__(self, environment:Environment, **kwargs) -> float:
        """
        Calculate the dead end penalty.

        Args:
            environment: The environment to calculate the penalty for.
            action: The action taken in the environment.

        Returns:
            The calculated penalty.
        """
        visited_nodes = kwargs.get('visited_nodes', List[int])
        current_node = kwargs.get('current_node', int)
        environment = kwargs.get('environment', Environment)
        
        no_moves = all(v in visited_nodes for v,_ in environment.adjacency_matrix[current_node])
        if no_moves:
            environment.states[-1].terminated = True 
            return self.penalty
        return 0.0
    
class WaitTimePenalty(Penalty):
    """
    Penalty for waiting at a traffic light in the environment.
    """

    def __init__(self, name: str, penalty:float=None):
        super().__init__(name, penalty)
        

    def __call__(self, **kwargs) -> float:
        """
        Calculate the wait time penalty.

        Args:
            environment: The environment to calculate the penalty for.
            action: The action taken in the environment.

        Returns:
            The calculated penalty.
        """
        action = kwargs.get('action', int)
        environment = kwargs.get('environment', Environment)
        
        if environment.states[-1].static_data.x[:,0][action] and not environment.states[-1].dynamic_data.x[:,0][action]:
            return -(environment.states[-1].dynamic_data.x[:,1][action])
        return 0.0

