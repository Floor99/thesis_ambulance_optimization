

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
