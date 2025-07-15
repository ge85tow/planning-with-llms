import os
import torch
from datasets import load_from_disk
from verl.algorithms 
from verl.algorithms import GRPO
from verl.envs.base_env import BaseEnv  # or your custom env class
from verl.trainer import RLTrainer
from verl.models import load_llm_with_peft
import yaml

from verl_grpo_reward import format_reward, plan_reward

# Load config
with open("verl_grpo_config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

# Model and Tokenizer: LoRA/PEFT setup with Gemma-3-12b-IT
# This function assumes you have a PEFT config file or dict as in your original code.
peft_args = cfg["peft"]
model, tokenizer = load_llm_with_peft(
    model_name="google/gemma-3-12b-it",
    peft_config=peft_args,
    cache_dir=cfg.get("cache_dir", None),
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

# Load Dataset (HuggingFace style)
def load_data(n, split):
    data_dir = f"/srv/chawak/planning-with-llms/data/{n}_blocks"
    data_path = f"{data_dir}/GRPO_systhink_tokenized_dataset/{split}"
    return load_from_disk(data_path)

# Define Environment Wrapper if needed (stub example)
class BlocksworldEnv(BaseEnv):
    def __init__(self, *args, **kwargs):
        # Initialize with your custom logic, if needed
        pass

    def reset(self):
        # Implement environment reset
        pass

    def step(self, action):
        # Implement environment step
        pass

    # Add other methods as needed for VERL RL loop

# Training Arguments
train_args = cfg["training"]
env = BlocksworldEnv()  # Stub; replace with your actual environment usage

# Load data
train_data = load_data(n=3, split="train")
sample_size = train_args.get("sample_size", 15)
train_data = train_data.select(range(sample_size))

# Prepare the GRPO algorithm
grpo = GRPO(
    model=model,
    env=env,
    reward_funcs=[format_reward, plan_reward],
    **train_args  # Pass other hyperparameters
)

# RL Trainer
trainer = RLTrainer(
    algorithm=grpo,
    train_dataset=train_data,
    tokenizer=tokenizer,
)

# Start training
trainer.train()

# Optionally, log metrics and save results
print("\n🔍 GPU Memory Summary after training:\n")
print(torch.cuda.memory_summary())