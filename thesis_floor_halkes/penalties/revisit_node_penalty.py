from thesis_floor_halkes.penalties.base import Penalty, Bonus


class RevisitNodePenalty(Penalty):
    """
    Penalty for revisiting a node in the environment.
    """

    def __init__(self, name: str):
        super().__init__(name)

    def __call__(self, environment, penalty: float, action) -> float:
        """
        Calculate the revisit node penalty.

        Args:
            environment: The environment to calculate the penalty for.
            action: The action taken in the environment.

        Returns:
            The calculated penalty.
        """
        if action in environment.visited_nodes:
            return penalty
        return 0.0

class PenaltyPerStep(Penalty):
    """
    Penalty for each step taken in the environment.
    """

    def __init__(self, name: str):
        super().__init__(name)

    def __call__(self, environment, penalty: float, action=None) -> float:
        """
        Calculate the penalty per step.

        Args:
            environment: The environment to calculate the penalty for.
            action: The action taken in the environment.

        Returns:
            The calculated penalty.
        """
        return penalty

class GoalBonus(Bonus):
    """
    Bonus for reaching the goal in the environment.
    """

    def __init__(self, name: str):
        super().__init__(name)

    def __call__(self, environment, bonus: float, action=None) -> float:
        """
        Calculate the goal bonus.

        Args:
            environment: The environment to calculate the bonus for.
            action: The action taken in the environment.

        Returns:
            The calculated bonus.
        """
        if environment.current_node == environment.goal_node:
            return bonus
        return 0.0

class DeadEndPenalty(Penalty):
    """
    Penalty for reaching a dead end in the environment.
    """

    def __init__(self, name: str):
        super().__init__(name)

    def __call__(self, environment, penalty: float, action=None) -> float:
        """
        Calculate the dead end penalty.

        Args:
            environment: The environment to calculate the penalty for.
            action: The action taken in the environment.

        Returns:
            The calculated penalty.
        """
        no_moves = all(v in environment.visited for v,_ in environment.adjacency_matrix[environment.current_node])
        if no_moves:
            environment.done = True 
            return penalty
        return 0.0
    
class WaitTimePenalty(Penalty):
    """
    Penalty for waiting at a traffic light in the environment.
    """

    def __init__(self, name: str):
        super().__init__(name)
        

    def __call__(self, environment, action, penalty: float, weight=1) -> float:
        """
        Calculate the wait time penalty.

        Args:
            environment: The environment to calculate the penalty for.
            action: The action taken in the environment.

        Returns:
            The calculated penalty.
        """
        if environment.traffic_lights[action] and not environment.light_status[action]:
            return penalty * weight
        return 0.0

