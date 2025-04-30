import torch
from torch_geometric.data import Data

class AmbulanceEnv:
    """
    Environment for ambulance routing problem.
    Constraints:
      - Only direct neighbors of current node can be chosen
      - Each node can be visited at most once per episode
    Node features:
      - data.x[:,0]: traffic_light (0/1)
      - data.x[:,1]: initial waiting_time (sec)
    Edge features:
      - data.edge_attr[:,0]: length (m)
      - data.edge_attr[:,1]: max_speed (km/h)
    Reward: negative (travel_time + wait_time)
    """
    def __init__(self, data: Data, start_node: int, end_node: int, goal_bonus: float, dead_end_penalty: float):
        # Store graph
        self.data = data
        self.start_node = start_node
        self.end_node = end_node
        self.num_nodes = data.num_nodes

        # Static and dynamic node features
        self.traffic_lights = data.x[:, 0].bool()
        self.initial_waiting = data.x[:, 1].clone()
        self.light_status   = data.x[:, 1].bool()  # True if green, False if red

        # Reward parameters
        self.goal_bonus = goal_bonus
        self.dead_end_penalty = dead_end_penalty
        
        # Precompute travel times per edge
        lengths = data.edge_attr[:, 0]
        speeds_kmh = data.edge_attr[:, 1]
        speeds_ms = speeds_kmh / 3.6
        self.edge_travel_times = lengths / speeds_ms

        # Adjacency list: node -> [(neighbor, edge_idx), ...]
        self.adj = {i: [] for i in range(self.num_nodes)}
        for idx, (u, v) in enumerate(data.edge_index.t().tolist()):
            self.adj[u].append((v, idx))

        self.reset()

    def reset(self):
        """
        Reset environment state.
        Returns initial observation.
        """
        self.current_node = self.start_node
        self.done = False
        # Dynamic waits
        self.waiting_times = self.initial_waiting.clone()
        # Track visited nodes
        self.visited = {self.start_node}
        return self._get_obs()

    def _get_obs(self):
        """
        Return (obs_data, current_node) carrying all three static node features:
        [is_traffic_light, light_status, waiting_time]
        """
        # 1) Forward the complete feature matrix as-is
        obs_data = Data(
            x=self.data.x.clone(),          # shape [N×3]
            edge_index=self.data.edge_index,
            edge_attr=self.data.edge_attr
        )
        # 2) Attach valid actions (neighbors not yet visited)
        obs_data.valid_actions = [
            v for v, _ in self.adj[self.current_node]
            if v not in self.visited
        ]
        return obs_data, self.current_node


    def step(self, action: int):
        """
        Move to a neighbor node. Cannot revisit a node or choose non-neighbor.
        Returns: (obs, reward, done, info)
        """
        if self.done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        # Validity checks
        neighbors = [v for v, _ in self.adj[self.current_node]]
        if action not in neighbors:
            raise ValueError(f"Invalid action {action}. Not a neighbor of {self.current_node}.")
        if action in self.visited:
            raise ValueError(f"Invalid action {action}. Node already visited.")

        # Compute reward
        edge_idx = next(idx for (v, idx) in self.adj[self.current_node] if v == action)
        travel_t = self.edge_travel_times[edge_idx]
        
        # Compute waiting: only if red light; if green skip waiting
        if self.traffic_lights[action] == 1 and self.light_status[action] == 0:             # there is a traffic light and status is red
            wait_t = float(self.data.x[action, 2].item())
            # print(f"Waiting time at {action}: {wait_t:.2f}s, status: {'green' if self.light_status[action] else 'red'}")
        else:
            wait_t = 0.0
            # print(f"No traffic light at {action} or traffic light is green, no waiting time.")

        total_time = travel_t + wait_t
        reward = - total_time
        # print(f" reward of total travel time = {reward:.2f} ")
        
        # Update state
        self.current_node = action
        self.visited.add(action)
        # self.waiting_times[action] = 0.0
        
        # Check terminal conditions
        if self.current_node == self.end_node:
            self.done = True
            reward += self.goal_bonus
            # print(f"Episode ended: reached end node {self.end_node}.")
        else:
            # if no further valid moves, end episode
            no_moves = all(v in self.visited for v, _ in self.adj[self.current_node])
            if no_moves:
                self.done = True
                reward -= self.dead_end_penalty
                # print(f"Episode ended: no valid moves from {self.current_node}.")

        return self._get_obs(), reward, self.done, {}

    def get_valid_actions(self):
        """
        Return neighbors of current_node not visited yet.
        """
        return [v for v, _ in self.adj[self.current_node] if v not in self.visited]


###################################################### DYNAMIC ENV ######################################################
import torch
from torch_geometric.data import Data

class AmbulanceEnvDynamic:
    """
    Environment for ambulance routing with dynamic waiting times and light status.

    Static:
      - traffic_lights: bool vector [N] indicating where lights exist
      - edge_index: connectivity
      - edge_attr: [E×2] length, max_speed
    Dynamic (re-sampled each step):
      - light_status: bool vector [N] (0=red,1=green)
      - waiting_times: float vector [N] seconds

    Reward: negative (travel_time + wait_time), plus bonuses/penalties.
    """
    def __init__(
        self,
        data: Data,
        start_node: int,
        end_node: int,
        goal_bonus: float = 0.0,
        dead_end_penalty: float = 0.0,
        max_wait: float = 30.0
    ):
        # Store graph and parameters
        self.data = data
        self.start_node = start_node
        self.end_node = end_node
        self.num_nodes = data.num_nodes

        # Static node feature: where traffic lights exist
        self.traffic_lights = data.x[:, 0].bool()

        # Reward parameters
        self.goal_bonus = goal_bonus
        self.dead_end_penalty = dead_end_penalty
        self.max_wait = max_wait

        # Precompute static edge travel times
        lengths = data.edge_attr[:, 0]
        speeds_kmh = data.edge_attr[:, 1]
        speeds_ms = speeds_kmh / 3.6
        self.edge_travel_times = lengths / speeds_ms

        # Build adjacency for neighbor lookups
        self.adj = {i: [] for i in range(self.num_nodes)}
        for idx, (u, v) in enumerate(data.edge_index.t().tolist()):
            self.adj[u].append((v, idx))

        # Initialize dynamic state and visited set
        self.reset()

    def reset(self):
        """
        Reset to start node and sample initial dynamic features.
        Returns initial observation.
        """
        self.current_node = self.start_node
        self.done = False
        
        # Sample dynamic features fresh each episode
        dev = self.traffic_lights.device
        rand_bits = torch.randint(0, 2, (self.num_nodes,), dtype=torch.bool, device=dev)    # get random bits
        self.light_status = rand_bits & self.traffic_lights                                 # if there is a traffic light, set status from random bits
        self.waiting_times = torch.rand(self.num_nodes, device = dev) * self.max_wait       # set random waiting time
        
        # Reset visited
        self.visited = {self.start_node}
        return self._get_obs()

    def _get_obs(self):
        """
        Build observation with combined static+dynamic node features.
        Returns (obs_data, current_node).
        obs_data.x columns: [traffic_light, light_status, waiting_time]
        """
        x = torch.stack([
            self.traffic_lights.to(torch.float),        # static feature
            self.light_status.to(torch.float),          # dynamic feature
            self.waiting_times                          # dynamic feature
        ], dim=1)                                       # this constructs the feature matrix - shape is [Nx3] (3 is nr of features in this case )
        
        obs_data = Data(
            x=x,
            edge_index=self.data.edge_index,
            edge_attr=self.data.edge_attr
        )                                               # packs into a new Data object with node and edge information - Policy needs PyG Data object
        
        obs_data.valid_actions = [
            v for v, _ in self.adj[self.current_node]
            if v not in self.visited
        ]                                               # attach valid actions (neighbors not yet visited)
        
        return obs_data, self.current_node  

    def step(self, action: int):
        """
        Execute action, update dynamic features, compute reward.
        Returns (obs, reward, done, info).
        """
        if self.done:
            raise RuntimeError("Episode done; call reset() to restart.")
        # Validate
        neighbors = [v for v, _ in self.adj[self.current_node]]
        if action not in neighbors or action in self.visited:
            raise ValueError(f"Invalid action {action} from node {self.current_node}.")

        # Compute the travel time
        edge_idx = next(idx for (v, idx) in self.adj[self.current_node] if v == action)
        travel_t = self.edge_travel_times[edge_idx]
        # Add waiting time if traffic light is red, when the traffic light is green there is no waiting time
        if self.traffic_lights[action] and not self.light_status[action]:
            wait_t = float(self.waiting_times[action].item())
        else:
            wait_t = 0.0

        reward = - (travel_t + wait_t)

        # Move to next node and mark the last node as visited
        self.current_node = action
        self.visited.add(action)

        # Re-sample dynamic features for next step
        dev = self.traffic_lights.device
        rand_bits = torch.randint(0, 2, (self.num_nodes,), dtype=torch.bool, device=dev)        # random bit
        self.light_status   = rand_bits & self.traffic_lights                                   # if there is a traffic light, set status from random bits
        self.waiting_times  = torch.rand(self.num_nodes, device=dev) * self.max_wait            # set random waiting time

        # Terminal checks: if end node reached goal it gets a bonus, otherwise it gets a penalty if it doesn't reach the end node
        if self.current_node == self.end_node:
            self.done = True
            reward += self.goal_bonus
        else:
            no_moves = all(v in self.visited for v, _ in self.adj[self.current_node])
            if no_moves:
                self.done = True
                reward -= self.dead_end_penalty

        return self._get_obs(), reward, self.done, {}

    def get_valid_actions(self):
        """
        Return neighbors of current_node not yet visited.
        """
        return [v for v, _ in self.adj[self.current_node] if v not in self.visited]
