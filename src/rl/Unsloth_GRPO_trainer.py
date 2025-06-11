import sys
sys.path.append("/srv/chawak/planning-with-llms/src/")

from huggingface_hub import login
from datasets import Dataset
login(token="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")

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'

from rl import GRPO_utils
import regex as re

#----------------------------- reward functions ----------------------------- 
format_pattern=r".*?\[PLAN\](.*?)\[PLAN END\]"
#<think>(.*?)<\/think>
format_pattern=re.compile(format_pattern, re.DOTALL)

#completions is a list of completion responses from model
def format_reward(completions, **kwargs) -> list[float]:
    
    scores=[]
    responses= [completion for completion in completions]
    
    for response in responses:
        print(f'In format reward, model response is: {response}')

        score=0
        if format_pattern.match(response):
            score = 10
        scores.append(score)

        print(f"Format reward for this response is: {score}")
    return scores

#completions is a list of completion responses from model
def plan_reward(completions,init, goal, gold_plan, **kwargs) -> list[float]:
    
    responses = [completion for completion in completions]
    init_list=init
    goal_list=goal
    gold_plan_list=gold_plan

    scores=[]
    scores=GRPO_utils.responses_scores(response_list=responses, init_list=init_list, goal_list=goal_list, gold_plan_list=gold_plan_list)           
    
    return scores

#----------------------------- GRPO set-up ----------------------------- 
from trl import GRPOConfig, GRPOTrainer
import yaml
from datasets import concatenate_datasets

#get config file
with open("/srv/chawak/planning-with-llms/src/rl/config.yaml", "r") as f:
    cfg=yaml.safe_load(f)

#set llm objects
peft_model=GRPO_utils.peft_model
tokenizer=GRPO_utils.tokenizer

import wandb
wandb.login(key="76161834310fa3386c0a678d4e30e18138446786")
wandb.init(project="GRPO-reboost")

def get_train_args():
    training_args=GRPOConfig(
        output_dir=cfg['training']['output_dir'],
        #overwrite_output_dir=False, #resume from checkpoint
        #resume_from_checkpoint=checkpoint_path, #resume from checkpoint
        num_train_epochs=int(cfg['training']['num_train_epochs']),
        per_device_train_batch_size=int(cfg['training']['per_device_train_batch_size']),
        gradient_accumulation_steps=int(cfg['training']['gradient_accumulation_steps']),
        learning_rate=float(cfg['training']['learning_rate']),
        adam_beta1=float(cfg['training']['adam_beta1']),
        adam_beta2=float(cfg['training']['adam_beta2']),
        weight_decay=float(cfg['training']['weight_decay']),
        warmup_ratio=float(cfg['training']['warmup_ratio']),
        lr_scheduler_type=cfg['training']['lr_scheduler_type'],
        optim=cfg['training']['optim'],
        num_generations=int(cfg['training']['num_generations']),
        max_completion_length=int(cfg['training']['max_completion_length']),
        logging_strategy=cfg['training']['logging_strategy'],
        save_strategy=cfg['training']['save_strategy']
    )
    return training_args

def train(train_data,model=peft_model):

    training_args=get_train_args()
    trainer = GRPOTrainer(
        model = model,
        reward_funcs=[
            format_reward,
            plan_reward,
        ],
        args = training_args,
        train_dataset = train_data,
        processing_class = tokenizer
    )
    trainer.train()

def main():
    
    #fetch train and eval data
    three_train = GRPO_utils.GRPO_load_tokenized_data(3)
    # four_train = GRPO_utils.GRPO_load_tokenized_data(4)
    train_data=three_train.select(range(10))

    #combine three and four block dataset to one
    # train_data=concatenate_datasets([three_train,four_train])

    train(train_data=train_data)

main()


