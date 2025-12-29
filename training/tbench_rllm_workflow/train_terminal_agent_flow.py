import hydra


from rllm.data.dataset import DatasetRegistry
from rllm.rewards.countdown_reward import countdown_reward_fn
from rllm.trainer.agent_trainer import AgentTrainer
from terminal_agent_flow import TerminalAgentWorkflow

@hydra.main(config_path="pkg://rllm.trainer.config", config_name="agent_ppo_trainer", version_base=None)
def main(config):
    train_dataset = DatasetRegistry.load_dataset("terminaldata", "train")
    test_dataset = DatasetRegistry.load_dataset("terminaldata", "test")

    # TODO: 
    # 1. <workflow_args>: what parameter does workflow_class need?
    # 2. <config>: set up in training script
    # 3. 

    max_steps = 50
    global_agent_timeout_sec = 600.0
    trainer = AgentTrainer(
        workflow_class=TerminalAgentWorkflow,
        workflow_args={
            "model_name": model_name,
            "env_args": {
                "cleanup": True,
            },
            "max_steps": max_steps,
            "global_agent_timeout_sec": global_agent_timeout_sec,
        },
        config=config,
        train_dataset=train_dataset,
        val_dataset=test_dataset,
    )
    trainer.train()

if __name__ == "__main__":
    main()
