import torch 

from model_dynamic_attention import PolicyNetworkGATDynamicAttention
from reinforce_baselines import NoBaseline, ExponentialBaseline

class AmbulanceAgent:
    def __init__(self, policy: PolicyNetworkGATDynamicAttention, baseline=None, lr: float = 1e-3, gamma: float = 0.99):
        self.policy    = policy
        self.baseline  = baseline or NoBaseline()
        self.optimizer = torch.optim.Adam(list(policy.parameters()) + baseline.get_learnable_parameters(), lr=lr)
        self.gamma     = gamma              # discount factor
        self.log_probs = []
        self.rewards   = []
    
    def select_action(self, obs):
        data, current_node = obs                            # obs: tuple (data(=node and edge features+valid actions), current_node)
        valid_actions = data.valid_actions                  # Extract the environment's action mask - valid actions            
        action, log_prob = self.policy(data, current_node, valid_actions)   # Policy network picks an action and returns its log probability (forward pass)
        self.log_probs.append(log_prob)                     # store log probs to compute gradients later in finish_episode
        return action, log_prob

    def store_reward(self, reward: float):
        self.rewards.append(reward)

    def finish_episode(self):
        # 1) compute raw discounted returns
        R = 0
        returns = []
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        device = self.log_probs[0].device if self.log_probs else torch.device('cpu')
        returns = torch.tensor(returns, device=device)

        # 2) evaluate baseline(s)
        # baselines, loss_b = self.baseline.eval(self.collected_states, returns)    # when adding critic baseline, use collected_states
        baselines, loss_b = self.baseline.eval(None, returns)                       # loss_b Teaches baseline how to estimate returns


        # 3) compute advantages
        advantages = returns - baselines

        # 4) (optional) normalize advantages for stability
        # if advantages.numel() > 1:
        #     advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        # 5) actor loss
        log_probs_tensor = torch.stack(self.log_probs)
        actor_loss = -(log_probs_tensor * advantages.detach()).mean()      # policy gradient loss, which updates policy network (actor) to make better decisions

        # 6) total loss = actor + baseline loss
        loss = actor_loss + loss_b                              # loss_b only non-zero when you’re using a learnable baseline

        # 7) backprop
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 8) reset
        self.log_probs.clear()
        self.rewards.clear()
        # self.collected_states.clear()
        
        return actor_loss, loss_b