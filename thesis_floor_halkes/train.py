# # train_dynamic.py
# # Training script for dynamic GAT-based ambulance routing with static caching

# import torch
# import matplotlib
# matplotlib.use('Agg')  # headless backend
# import matplotlib.pyplot as plt
# from torch_geometric.loader import DataLoader
# from graph_data import RandomGraphDataset
# from environment import AmbulanceEnvDynamic
# # from model_dynamic import PolicyNetworkGATDynamic, PolicyNetworkGATDynamicAttention
# from agent import AmbulanceAgent
# from reinforce_baselines import ExponentialBaseline

# # Hyperparameters
# in_static        = 1    # traffic_light only
# in_dyn           = 2    # light_status + waiting_time
# hidden_size      = 128
# static_layers    = 2
# static_heads     = 4
# dyn_heads        = 1
# dropout          = 0.2
# edge_attr_dim    = 2
# lr               = 1e-3
# gamma            = 0.99
# num_epochs       = 1
# num_graphs       = 5
# batch_size       = 1
# max_steps        = 40
# goal_bonus       = 10.0
# dead_end_penalty = 5.0
# max_wait         = 30.0

# # Device setup
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # Dataset and DataLoader
# dataset = RandomGraphDataset(
#     num_graphs=num_graphs,
#     min_nodes=5,
#     max_nodes=20,
#     min_prob=0.1,
#     max_prob=0.5,
#     max_wait=max_wait,
#     min_length=100.0,
#     max_length=1000.0,
#     min_speed=30.0,
#     max_speed=100.0
# )
# loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# # Model, baseline, and agent
# # policy = PolicyNetworkGATDynamic(
# #     in_static=in_static,
# #     in_dyn=in_dyn,
# #     hidden_size=hidden_size,
# #     static_layers=static_layers,
# #     static_heads=static_heads,
# #     dyn_heads=dyn_heads,
# #     dropout=dropout,
# #     edge_attr_dim=edge_attr_dim
# # ).to(device)

# from model_dynamic_attention import PolicyNetworkGATDynamicAttention

# policy = PolicyNetworkGATDynamicAttention(in_static=in_static,
#     in_dyn=in_dyn,
#     hidden_size=hidden_size,
#     static_layers=static_layers,
#     static_heads=static_heads,
#     dyn_heads=dyn_heads,
#     dropout=dropout,
#     edge_attr_dim=edge_attr_dim
# ).to(device)

# baseline = ExponentialBaseline(beta=0.99)
# agent = AmbulanceAgent(policy, baseline=baseline, lr=lr, gamma=gamma)

# # Tracking
# all_returns = []
# best_return = float('-inf')
# best_route  = None

# for epoch in range(1, num_epochs + 1):
#     print(f"=== Epoch {epoch}/{num_epochs} ===")
#     for batch_idx, batch in enumerate(loader, start=1):
#         for data in batch.to_data_list():
#             # Move graph to device
#             data = data.to(device)

#             # Random start and end
#             N = data.num_nodes
#             start = torch.randint(0, N, (1,)).item()
#             end   = torch.randint(0, N - 1, (1,)).item()
#             while end == start:
#                 end = torch.randint(0, N - 1, (1,)).item()

#             # Initialize dynamic environment
#             env = AmbulanceEnvDynamic(
#                 data.clone(),
#                 start_node=start,
#                 end_node=end,
#                 goal_bonus=goal_bonus,
#                 dead_end_penalty=dead_end_penalty,
#                 max_wait=max_wait
#             )

#             # Episode start: reset env & clear static cache
#             obs, curr = env.reset()
#                 # check new graphs
#             print(f"\n=== New Episode ===")
#             print(f"Graph has {data.num_nodes} nodes; start={start}, end={end}")
#             print("Edges:", data.edge_index.t().tolist())
#                 # check static features
#             static_ref = data.x[:, :in_static].clone()
#             print("Static is_light:", static_ref.tolist())
            
#             policy.clear_static_cache()
#             agent.log_probs.clear()
#             agent.rewards.clear()

#             route = [curr]
#             ep_return = 0.0
            
#                 # Print initial dynamic
#             dyn0 = obs.x[:, in_static:]
#             print(f" Step  0 dynamic features (status, wait): {dyn0.tolist()}")

#             # Rollout - steps
#             for t in range(max_steps):
#                 action, _ = agent.select_action((obs, curr))
#                 (obs, curr), reward, done, _ = env.step(action)
#                     # Check static is unchanged
#                 static_now = data.x[:, :in_static]
#                 assert torch.equal(static_ref, static_now), "Static features mutated!"

#                     # Print dynamic slice from the *new* obs_data
#                 dyn_now = obs.x[:, in_static:]
#                 print(f" Step {t:2d} dynamic features (status, wait): {dyn_now.tolist()}")
                
#                 agent.store_reward(float(reward))
#                 ep_return += float(reward)
#                 route.append(curr)
#                 if done:
#                     break

#             # Update policy
#             agent.finish_episode()
#             all_returns.append(ep_return)

#             # Track best
#             if curr == end and ep_return > best_return:
#                 best_return = ep_return
#                 best_route  = route.copy()

#         print(f"Batch {batch_idx}/{len(loader)} → best return {best_return:.2f} on route {best_route}")
    

# print(f"Training complete. Best return: {best_return:.2f}, route: {best_route}")

# # Plot returns
# plt.figure()
# plt.plot(all_returns)
# plt.xlabel('Episode')
# plt.ylabel('Return')
# plt.title('Dynamic-Graph Training Returns')
# plt.tight_layout()
# plt.savefig('dynamic_training_returns.png', dpi=300)
# print('Saved dynamic_training_returns.png')

import argparse
import torch
import matplotlib
matplotlib.use('Agg')  # headless backend
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader
from graph_data import RandomGraphDataset
from environment import AmbulanceEnvDynamic
from model_dynamic_attention import PolicyNetworkGATDynamicAttention
from agent import AmbulanceAgent
from reinforce_baselines import ExponentialBaseline

def train(
    in_static: int,
    in_dyn: int,
    hidden_size: int,
    static_layers: int,
    static_heads: int,
    dyn_heads: int,
    dropout: float,
    edge_attr_dim: int,
    lr: float,
    gamma: float,
    num_epochs: int,
    num_graphs: int,
    batch_size: int,
    max_steps: int,
    goal_bonus: float,
    dead_end_penalty: float,
    max_wait: float,
    save_path: str
):
    """
    Train the dynamic GAT-based ambulance routing policy.
    Saves best policy weights to `save_path` and plots training returns.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Dataset and DataLoader
    dataset = RandomGraphDataset(
        num_graphs=num_graphs,
        min_nodes=10,
        max_nodes=40,
        min_prob=0.1,
        max_prob=0.5,
        max_wait=max_wait,
        min_length=100.0,
        max_length=1000.0,
        min_speed=30.0,
        max_speed=100.0
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Model, baseline, and agent
    policy = PolicyNetworkGATDynamicAttention(
        in_static=in_static,
        in_dyn=in_dyn,
        hidden_size=hidden_size,
        static_layers=static_layers,
        static_heads=static_heads,
        dyn_heads=dyn_heads,
        dropout=dropout,
        edge_attr_dim=edge_attr_dim
    ).to(device)

    baseline = ExponentialBaseline(beta=0.99)
    agent = AmbulanceAgent(policy, baseline=baseline, lr=lr, gamma=gamma)

    all_returns = []
    actor_losses = []
    baseline_losses = []
    best_return = float('-inf')
    best_route  = None

    for epoch in range(1, num_epochs + 1):
        print(f"=== Epoch {epoch}/{num_epochs} ===")
        for batch_idx, batch in enumerate(loader, start=1):
            for data in batch.to_data_list():
                data = data.to(device)

                # Random start/end
                N = data.num_nodes
                start = torch.randint(0, N, (1,)).item()
                end   = torch.randint(0, N - 1, (1,)).item()
                while end == start:
                    end = torch.randint(0, N - 1, (1,)).item()

                # Environment
                env = AmbulanceEnvDynamic(
                    data.clone(),
                    start_node=start,
                    end_node=end,
                    goal_bonus=goal_bonus,
                    dead_end_penalty=dead_end_penalty,
                    max_wait=max_wait
                )

                # Reset
                (obs, curr) = env.reset()
                policy.clear_static_cache()
                agent.log_probs.clear()
                agent.rewards.clear()

                route = [curr]
                ep_return = 0.0

                # Rollout
                for t in range(max_steps):
                    action, _ = agent.select_action((obs, curr))
                    (obs, curr), reward, done, _ = env.step(action)
                    reward = reward.item() if torch.is_tensor(reward) else float(reward)
                    agent.store_reward(reward)
                    ep_return += reward
                    route.append(curr)
                    if done:
                        break

                # agent.finish_episode()
                actor_loss, loss_b = agent.finish_episode()
                actor_losses.append(float(actor_loss))
                baseline_losses.append(float(loss_b))
                all_returns.append(float(ep_return))

                # Track best
                if curr == end and ep_return > best_return:
                    best_return = ep_return
                    best_route  = route.copy()

            print(f"Batch {batch_idx}/{len(loader)} → best return {best_return:.2f} on route {best_route}")

    print(f"Training complete. Best return: {best_return:.2f}, route: {best_route}")

    # Save policy weights
    torch.save(policy.state_dict(), save_path)
    print(f"Saved best policy weights to {save_path}")

    # Plot returns
    plt.figure()
    plt.plot(all_returns)
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.title('Dynamic-Graph Training Returns')
    plt.tight_layout()
    plt.savefig('thesis_floor_halkes/plots/dynamic_training_returns.png', dpi=300)
    print('Saved dynamic_training_returns.png')
    
        # Plot losses
    plt.figure()
    plt.plot(actor_losses, label='Actor Loss')
    plt.plot(baseline_losses, label='Baseline Loss')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.title('Actor and Baseline Loss per Episode')
    plt.legend()
    plt.savefig('thesis_floor_halkes/plots/dynamic_training_losses.png', dpi=300)
    print('Saved dynamic_training_losses.png')

def parse_args():
    parser = argparse.ArgumentParser("Train dynamic GAT routing policy")
    parser.add_argument('--epochs',    type=int,   default=1000)
    parser.add_argument('--graphs',    type=int,   default=512)
    parser.add_argument('--batch',     type=int,   default=16)
    parser.add_argument('--steps',     type=int,   default=50)
    parser.add_argument('--save',      type=str,   default='thesis_floor_halkes/policy/best_policy.pth')
    parser.add_argument('--lr',        type=float, default=1e-3)
    parser.add_argument('--gamma',     type=float, default=0.99)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(
        in_static=1,
        in_dyn=2,
        hidden_size=128,
        static_layers=2,
        static_heads=4,
        dyn_heads=1,
        dropout=0.2,
        edge_attr_dim=2,
        lr=args.lr,
        gamma=args.gamma,
        num_epochs=args.epochs,
        num_graphs=args.graphs,
        batch_size=args.batch,
        max_steps=args.steps,
        goal_bonus=10.0,
        dead_end_penalty=5.0,
        max_wait=30.0,
        save_path=args.save
    )
