#import unsloth
import sys
sys.path.append("/srv/chawak/planning-with-llms/src/")

from huggingface_hub import login
login(token="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")

import os
os.environ["WANDB_MODE"] = "disabled"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from rl import GRPO_utils
from shared import llm_utils
import regex as re

#reward functions
format_pattern=r"<think>(.*?)<\/think>.*?\[PLAN\](.*?)\[PLAN END\]"
format_pattern=re.compile(format_pattern, re.DOTALL)

def format_reward(prompts,completions, **kwargs) -> list[float]:
    #responses= [completion[0]["content"] for completion in completions]
    responses= [completion[0] for completion in completions]
    return [0.0 if not format_pattern.match(response) else 10.0 for response in responses]

def plan_reward(prompts,completions,init, goal, gold_plan, **kwargs) -> list[float]:
    #responses= [completion[0]["content"] for completion in completions]
    responses= [completion[0] for completion in completions]
    scores=[]
    # for response in responses:
        
    print(f"Response in plan reward is: {responses}")
    score=0
    score=GRPO_utils.response_score(response=responses, init=init, goal=goal, gold_plan=gold_plan)
    scores.append(score)            
    return scores

#----------------------------- GRPO set-up ----------------------------- 

#from unsloth import FastLanguageModel

from transformers import DataCollatorWithPadding
from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig, get_peft_model
import yaml,torch
from datasets import concatenate_datasets

#get config file
with open("/srv/chawak/planning-with-llms/src/rl/config.yaml", "r") as f:
    cfg=yaml.safe_load(f)

#get model to be trained
base_model,tokenizer=llm_utils.get_model_tokenizer()
base_model.gradient_checkpointing_enable()

#LORA config 
peft_args=LoraConfig(
    r=int(cfg['peft']['r']),
    lora_alpha=int(cfg['peft']['lora_alpha']),
    #lora_dropout=float(cfg['peft']['lora_dropout']),
    task_type=cfg['peft']['task_type'],
    target_modules=list(cfg['peft']['target_modules'])
)

peft_model=get_peft_model(base_model,peft_args)

#set environment variables
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'

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
    three_train, three_eval = llm_utils.GRPO_load_tokenized_data(3)
    four_train, four_eval = llm_utils.GRPO_load_tokenized_data(4)

    #combine three and four block dataset to one
    train_data=concatenate_datasets([three_train,four_train])
    eval_data= concatenate_datasets([three_eval,four_eval])

    train(train_data=train_data)

main()


#unsloth loading of model
'''gemmamodel=FastLanguageModel.from_pretrained(
    model_name="google/gemma-3-12b-it",
    load_in_4bit=False,
    fast_inference=True,
    max_lora_rank=64
    )
lora_model = FastLanguageModel.get_peft_model(
    gemmamodel,
    r=64,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],  # Remove QKVO if out of memory
    lora_alpha=64,
    use_gradient_checkpointing="unsloth",  # Enable long context finetuning
    random_state=3407,
) '''