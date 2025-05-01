from thesis_floor_halkes.penalties.base import Penalty


class RevisitNodePenalty(Penalty):
    """
    Penalty for revisiting a node in the environment.
    """

    def __init__(self, name: str, penalty: float):
        super().__init__(name)
        self.penalty = penalty

    def __call__(self, environment, action=None) -> float:
        """
        Calculate the revisit node penalty.

        Args:
            environment: The environment to calculate the penalty for.
            action: The action taken in the environment.

        Returns:
            The calculated penalty.
        """
        if action in environment.visited_nodes:
            return self.penalty
        return 0.0
