from abc import ABC, abstractmethod
from torch_geometric.data import Data


class StaticFeatureGetter(ABC):
    """
    Abstract base class for static feature getters.
    """
    @abstractmethod
    def get_static_features(self):
        """
        Abstract method to get static features.
        """
        pass

class StaticFeatureFromListGetter(StaticFeatureGetter):
    """
    Static feature getter that retrieves features from a list.
    """
    def __init__(self, static_features: list[Data]):
        """
        Initialize the static feature getter with a list of static features.

        Args:
            static_features: List of Data objects with static features.
        """
        pass
        