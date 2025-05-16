


import mlflow


def log_episode_artifacts(
    episode_info, 
    epoch, 
    graph_idx, 
    fig
):
    graph_folder = f"epoch_{epoch:03}/graph_{graph_idx:02}"
    mlflow.log_dict(episode_info, f"{graph_folder}/info.json")
    mlflow.log_figure(fig, f"{graph_folder}/graph.png")
    
def log_agent_checkpoint(agent, epoch):
    mlflow.pytorch.log_state_dict({
        "static_encoder": agent.static_encoder.state_dict(),
        "dynamic_encoder": agent.dynamic_encoder.state_dict(),
        "decoder": agent.decoder.state_dict(),
        "baseline": agent.baseline.state_dict()
    }, artifact_path=f"epoch_{epoch:03}/agent")

def log_full_models(agent):
    mlflow.pytorch.log_model(agent.static_encoder, "static_encoder_model")
    mlflow.pytorch.log_model(agent.dynamic_encoder, "dynamic_encoder_model")
    mlflow.pytorch.log_model(agent.decoder, "decoder_model")
    mlflow.pytorch.log_model(agent.baseline, "baseline_model")
    
def log_gradient_norms(module, module_name, step_id):
    for name, param in module.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            mlflow.log_metric(f"grad_norm/{module_name}/{name}", grad_norm, step=step_id)