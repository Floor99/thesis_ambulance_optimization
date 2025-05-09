from typing import List

import torch
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
    
class CloserToGoalBonus(Bonus):
    """
    Bonus for every step getting closer to the goal in the environment, based on Euclidean distance.
    """
    def __init__(self, name: str, bonus:float, data, end_node, scaled:bool = False
                 ):
        super().__init__(name, bonus)
        self.pos = data.pos
        self.goal_pos = data.pos[end_node]
        self.scaled = scaled

    def __call__(self, **kwargs) -> float:
        """
        Calculate the closer to goal bonus, based on Euclidean distance to goal each step.

        Args:
            environment: The environment to calculate the bonus for.
            action: The action taken in the environment.

        Returns:
            The calculated bonus.
        """
        prev_idx = kwargs.get('prev_node', int)
        curr_idx = kwargs.get('current_node', int)
        
        prev_pos = self.pos[prev_idx]
        curr_pos = self.pos[curr_idx]
        
        d_prev = torch.dist(prev_pos, self.goal_pos)
        dcurr = torch.dist(curr_pos, self.goal_pos)
        delta = (d_prev - dcurr).item()
        
        if delta <= 0:
            return 0.0

        return self.bonus * (delta if self.scaled else 1.0)
    
class HigherSpeedBonus(Bonus):
    """
    Bonus for roads with a higher speed limit in the environment.
    """
    def __init__(self, name, bonus):
        super().__init__(name, bonus)
        
        pass
    
class MoreLanesBonus(Bonus):
    """
    Bonus for roads with more lanes in the environment.
    """
    def __init__(self, name, bonus):
        super().__init__(name, bonus)
        
        pass
    
