import argparse
import torch
import matplotlib
matplotlib.use('Agg')  # headless backend
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader
from graph_data import RandomGraphDataset, GraphGenerator, TrafficGraphSet
from environment import AmbulanceEnvDynamic
from model_dynamic_attention import PolicyNetworkGATDynamicAttention
from agent import AmbulanceAgent
from reinforce_baselines import ExponentialBaseline, CriticBaseline
from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np
from critic_network import CriticNetwork

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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    writer = SummaryWriter(log_dir=os.path.join('runs', 'reinforce_dynamic'))

    # Generate base graph for fixed-layout phase
    graph_gen = GraphGenerator(num_nodes=10, edge_prob=0.4, max_wait=max_wait)
    base_graph = graph_gen.generate()

    # Create dynamic features (fixed layout, varying time steps)
    time_blocks = 4000
    wait_tensor = torch.rand(base_graph.num_nodes, time_blocks) * max_wait
    status_tensor = torch.randint(0, 2, (base_graph.num_nodes, time_blocks), dtype=torch.float)

    dataset = TrafficGraphSet(base_graph, wait_tensor, status_tensor, time_blocks=time_blocks)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

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

    critic = CriticNetwork(
        in_static=in_static,
        in_dynamic=in_dyn,
        hidden_size=hidden_size,
        edge_attr_dim=edge_attr_dim
    ).to(device)

    baseline = CriticBaseline(critic)
    # baseline = ExponentialBaseline(beta=0.99).to(device)
    agent = AmbulanceAgent(policy, baseline=baseline, lr=lr, gamma=gamma)

    all_returns = []
    actor_losses = []
    baseline_losses = []
    best_return = float('-inf')
    best_route  = None
    global_step = 0

    for epoch in range(1, num_epochs + 1):
        print(f"=== Epoch {epoch}/{num_epochs} ===")
        epoch_returns = []

        # Optionally transition to random graphs after warmup phase
        # if epoch == num_epochs // 2:
        #     print("Switching to full random graphs!")
        #     dataset = RandomGraphDataset(
        #         num_graphs=num_graphs,
        #         min_nodes=10,
        #         max_nodes=40,
        #         min_prob=0.1,
        #         max_prob=0.5,
        #         max_wait=max_wait,
        #         min_length=100.0,
        #         max_length=1000.0,
        #         min_speed=30.0,
        #         max_speed=100.0
        #     )
        #     loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for batch_idx, batch in enumerate(loader, start=1):
            for data in batch.to_data_list():
                data = data.to(device)

                N = data.num_nodes
                start = torch.randint(0, N, (1,)).item()
                end   = torch.randint(0, N - 1, (1,)).item()
                while end == start:
                    end = torch.randint(0, N - 1, (1,)).item()

                env = AmbulanceEnvDynamic(
                    data.clone(),
                    start_node=start,
                    end_node=end,
                    goal_bonus=goal_bonus,
                    dead_end_penalty=dead_end_penalty,
                    max_wait=max_wait,
                    revisit_penalty=1.0,
                    step_penalty=0.1,
                    wait_weight=2.0,
                )

                (obs, curr) = env.reset()
                print(f"Graph has {data.num_nodes} nodes; start={start}, end={end}")
                print("Edges:", data.edge_index.t().tolist())
                policy.clear_static_cache()
                agent.log_probs.clear()
                agent.rewards.clear()

                route = [curr]
                ep_return = 0.0

                for t in range(max_steps):
                    action, _ = agent.select_action((obs, curr))
                    (obs, curr), reward, done, _ = env.step(action)
                    reward = reward.item() if torch.is_tensor(reward) else float(reward)
                    agent.store_reward(reward)
                    ep_return += reward
                    route.append(curr)
                    if done:
                        break

                actor_loss, loss_b = agent.finish_episode()
                actor_losses.append(float(actor_loss))
                baseline_losses.append(float(loss_b))
                all_returns.append(float(ep_return))
                epoch_returns.append(float(ep_return))

                writer.add_scalar('Return/episode', ep_return, global_step)
                writer.add_scalar('Loss/actor', actor_loss, global_step)
                writer.add_scalar('Loss/baseline', loss_b, global_step)
                writer.add_scalar('Episode/success', float(curr == end), global_step)
                writer.add_scalar('Loss/critic', loss_b, global_step)

                global_step += 1
                
                print(f"route {route} → return {ep_return:.2f} (actor loss: {actor_loss:.2f}, baseline loss: {loss_b:.2f})")
                
                if curr == end and ep_return > best_return:
                    best_return = ep_return
                    best_route  = route.copy()

            print(f"Batch {batch_idx}/{len(loader)} → best return {best_return:.2f} on route {best_route}")

        if epoch_returns:
            writer.add_scalar('Return/epoch_avg', np.mean(epoch_returns), epoch)

    print(f"Training complete. Best return: {best_return:.2f}, route: {best_route}")
    torch.save(policy.state_dict(), save_path)
    print(f"Saved best policy weights to {save_path}")

    plt.figure()
    plt.plot(all_returns)
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.title('Dynamic-Graph Training Returns')
    plt.tight_layout()
    plt.savefig('thesis_floor_halkes/plots/dynamic_training_returns.png', dpi=300)
    print('Saved dynamic_training_returns.png')

    plt.figure()
    plt.plot(actor_losses, label='Actor Loss')
    plt.plot(baseline_losses, label='Baseline Loss')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.title('Actor and Baseline Loss per Episode')
    plt.legend()
    plt.savefig('thesis_floor_halkes/plots/dynamic_training_losses.png', dpi=300)
    print('Saved dynamic_training_losses.png')

    writer.close()

def parse_args():
    parser = argparse.ArgumentParser("Train dynamic GAT routing policy")
    parser.add_argument('--epochs',    type=int,   default=50)
    parser.add_argument('--graphs',    type=int,   default=4000)
    parser.add_argument('--batch',     type=int,   default=64)
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
