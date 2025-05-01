from dataclasses import dataclass
from torch_geometric.data import Data


@dataclass
class State:
    data: Data
    current_node: int
    valid_actions: list[int] = None