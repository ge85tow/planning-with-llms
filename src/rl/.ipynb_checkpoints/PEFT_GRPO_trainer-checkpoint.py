import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'

# import warnings
# warnings.filterwarnings("ignore")
# import transformers
# transformers.logging.set_verbosity_error()

# import torch
# torch._dynamo.disable()  #debug for logging error

import sys
sys.path.append("/home/user/planning-with-llms/src/")

from huggingface_hub import login
from datasets import Dataset
login(token="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")


import regex as re
import torch

from rl import GRPO_utils
from rl import tracker
from rl import eval


#----------------------------- reward functions ----------------------------- 

gibberish=r"(.*)"
plan=r"\[PLAN\](.*?)\[PLAN END\]"
think=r"<think>(.*?)<\/think>"


def get_score(response):
    
    response=response.strip()
    score=0

    #define cases
    #case-1: no gibberish, no plan, yes think 
    pattern1=f"{think}"
    compiled1=re.compile(pattern1,re.DOTALL)

    #case-2: no gibberish, yes plan, no think
    pattern2=f"{plan}"
    compiled2=re.compile(pattern2,re.DOTALL)

    #case-3: no gibberish, yes plan, yes think
    pattern3=f"{think}\s*{plan}"
    compiled3=re.compile(pattern3,re.DOTALL)

    #case-6: yes gibberish, yes plan, yes think
    pattern6=f"{think}{gibberish}{plan}"
    compiled6=re.compile(pattern6,re.DOTALL)

    #ONLY think tag
    fullmatchthink=False
    if compiled1.fullmatch(response):
        score=7
        fullmatchthink=True
    if compiled1.search(response) and not fullmatchthink:
        score=5
    
    #ONLY plan tag
    fullmatchplan=False
    if compiled2.fullmatch(response):
        score=3
        fullmatchplan=True
    if compiled2.search(response) and not fullmatchplan:
        score=2
    
    #COMBO of both tags
    fullmatchcombo=False
    if compiled3.fullmatch(response):
        score=20
        fullmatchcombo=True
    if compiled6.search(response) and not fullmatchcombo:
        score=15

    return score

#completions is a list of completion responses from model
def format_reward(completions, **kwargs) -> list[float]:
    
    #memory debug
    torch.cuda.empty_cache()

    # if tracker.total_completions_seen==20:
    print(f'memory debug in reward/format_reward after {tracker.total_completions_seen} problems: {os.system("nvidia-smi")}')
    scores=[]
    responses= [completion for completion in completions]
    # print(f"Example propmt is {prompts[0]}")

    for response in responses:

        response=response.strip()
        print(f'{tracker.total_completions_seen}. In format reward, model response is: {repr(response)}')
        score=0

        #get format reward
        score=get_score(response)
        scores.append(score)

        print(f"Format reward for this response is: {score}")

    #for epoch evaluation and logging
    tracker.accumulate_format_reward(scores)

    return scores

#completions is a list of completion responses from model
def plan_reward(completions,init, goal, gold_plan, **kwargs) -> list[float]:
    
    #memory debug
    torch.cuda.empty_cache()

    # if tracker.total_completions_seen==20:
    #initialisations
    responses = [completion for completion in completions]
    init_list=init
    goal_list=goal
    gold_plan_list=gold_plan
    
    scores=[]
    #responses scores returns a tuple of lists for plan and bonus scores
    scores=GRPO_utils.responses_scores(response_list=responses, init_list=init_list, goal_list=goal_list, gold_plan_list=gold_plan_list)           
    plan_scores=scores[0]
    bonus_scores=scores[1]

    #for epoch evaluation and logging
    tracker.accumulate_plan_reward(plan_scores)
    tracker.accumulate_bonus_reward(bonus_scores)
    
    added_scores=[plan + bonus for plan,bonus in zip(plan_scores,bonus_scores)]
    
    
    return added_scores

#----------------------------- GRPO set-up ----------------------------- 
from trl import GRPOConfig, GRPOTrainer
import yaml
# from datasets import concatenate_datasets
from peft import LoraConfig, get_peft_model

#get config file
with open("/home/user/planning-with-llms/src/rl/config.yaml", "r") as f:
    cfg=yaml.safe_load(f)

#set llm objects
# peft_model=GRPO_utils.peft_model
# tokenizer=GRPO_utils.tokenizer
    
model=GRPO_utils.base_model
model.get_input_embeddings().weight.requires_grad = True
tokenizer=GRPO_utils.tokenizer

# #LORA config 
peft_args=LoraConfig(
    r=int(cfg['peft']['r']),
    lora_alpha=int(cfg['peft']['lora_alpha']),
    lora_dropout=float(cfg['peft']['lora_dropout']),
    task_type=cfg['peft']['task_type'],
    target_modules=list(cfg['peft']['target_modules']),
)

#debug for log error
# peft_model.generation_config.disable_compile = True
# peft_model.generate = torch._dynamo.disable(peft_model.generate)

import wandb
wandb.login(key="76161834310fa3386c0a678d4e30e18138446786")
wandb.init(project="GRPO-reboost")

def get_train_args(max_prompt_length):
    training_args=GRPOConfig(
        output_dir=cfg['training']['output_dir'],
        #overwrite_output_dir=False, #resume from checkpoint
        #resume_from_checkpoint=checkpoint_path, #resume from checkpoint
        num_train_epochs=int(cfg['training']['num_train_epochs']),
        per_device_train_batch_size=int(cfg['training']['per_device_train_batch_size']),
        per_device_eval_batch_size=int(cfg['training']['per_device_eval_batch_size']),
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
        save_strategy=cfg['training']['save_strategy'],
        temperature=cfg['generation']['temperature'],
        max_prompt_length=max_prompt_length,
        gradient_checkpointing=True, #memory debug
        #report_to=cfg['training']['report_to'] #keeping it w 
        #try for memory debug: disable_dropout=true
        #try for length based think trace: loss_type="dr_grpo"
    )
    return training_args

def train(train_data,model=model):

    #fetch max prompt length param for this data
    max_prompt_length=max([len(tokenized_prompt) for tokenized_prompt in train_data['input_ids']])
    #sanity-check
    print(f'MAX PROMPT LENGTH: {max_prompt_length}')

    training_args=get_train_args(max_prompt_length)
    model.generation_config.temperature = training_args.temperature
    trainer = GRPOTrainer(
        model = model,
        reward_funcs=[
            format_reward,
            plan_reward,
        ],
        args = training_args,
        train_dataset = train_data,
        processing_class = tokenizer,
        peft_config=peft_args
    )

    #sanity-check    
    print("🔍 Model after GRPOTrainer wrapping:")
    print(type(trainer.model))

    #check current generation temperature
    print(f"MODEL GENERATION CONFIG: {model.generation_config.to_dict()}")
    # print(f"DEBUG model in gradient required mode:")
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name)

    trainer.train()

    print("\n🔍 GPU Memory Summary after training:\n")
    print(torch.cuda.memory_summary())

def main(n,split):
    
    # #fetch train and eval data
    # #three_train = GRPO_utils.GRPO_load_tokenized_data(3)
    # #four_train = GRPO_utils.GRPO_load_tokenized_data(4)

    data=GRPO_utils.GRPO_load_tokenized_data(n,split)
    sample_size=1
    data=data.select(range(sample_size))
    print(f"Data-sample size is: {len(data)}")
    # #combine three and four block dataset to one
    # # train_data=concatenate_datasets([three_train,four_train])

    train(train_data=data)
    
    epoch_len=int(cfg['training']['num_generations'])*len(data)
    eval_df=eval.epoch_eval(tracker.format_scores,tracker.plan_scores,tracker.bonus_scores,epoch_len)
    path=cfg['training']['output_dir']
    eval_df.to_csv(path+'/metrics.csv')

    return data
main(n=3,split='train')