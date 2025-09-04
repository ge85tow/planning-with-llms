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

import pandas as pd
import regex as re
import torch

from rl import GRPO_utils
from rl import tracker
from rl import eval


#----------------------------- reward functions ----------------------------- 

lenfile='/home/user/planning-with-llms/results/rl/training/08_08/logs/lens.txt'

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

def get_completion_len(responses):
    lens=[]

    for r in responses:
        # print(f'response in get completion len : {r},')
        l=len(r)
        lens.append(l)
    return lens

def log_to_file(lens,total_completions_seen,path=lenfile):
    with open(path, "a") as f:
        step = total_completions_seen // float(cfg['training']['logging_steps'])
        lens_str = ",".join(str(l) for l in lens)
        content = f"{step},{lens_str}\n"
        f.write(content)
    return

#completions is a list of completion responses from model
def format_reward(completions, **kwargs) -> list[float]:
    
    #memory debug
    torch.cuda.empty_cache()

    # if tracker.total_completions_seen==20:
    print(f'memory debug in reward/format_reward after {tracker.total_completions_seen} problems: {os.system("nvidia-smi")}')
    scores=[]
    responses= [completion for completion in completions]
    
    lens=get_completion_len(responses)
    log_to_file(lens,tracker.total_completions_seen)

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
    terminate=scores[1]

    #for epoch evaluation and logging
    tracker.accumulate_plan_reward(plan_scores)
    tracker.accumulate_terminate(terminate)

    bonus_scores=[]
    for t in terminate:
        bonus_score=t[0]
        bonus_scores.append(bonus_score)
    
    added_scores=[plan + bonus for plan,bonus in zip(plan_scores,bonus_scores)]
    
    
    return added_scores

#----------------------------- GRPO set-up ----------------------------- 
from trl import GRPOConfig, GRPOTrainer
import yaml
from datasets import concatenate_datasets
from peft import LoraConfig, AdaLoraModel, AdaLoraConfig, get_peft_model

#get config file
with open("/home/user/planning-with-llms/src/rl/config.yaml", "r") as f:
    cfg=yaml.safe_load(f)

#set llm objects
# peft_model=GRPO_utils.peft_model
# tokenizer=GRPO_utils.tokenizer
    
model=GRPO_utils.base_model
model.get_input_embeddings().weight.requires_grad = True
tokenizer=GRPO_utils.tokenizer

peft_args=LoraConfig(
    peft_type=cfg['lora']['peft_type'],
    task_type=cfg['lora']['task_type'],
    r=int(cfg['lora']['r']),
    lora_alpha=int(cfg['lora']['lora_alpha']),
    lora_dropout=float(cfg['lora']['lora_dropout']),
    target_modules=list(cfg['lora']['target_modules']),
)

# Depreciated function:PEFT Model wrapping 
def _peft_model(type,sample_size):
    
    if type=='lora':
       peft_args=LoraConfig(
            peft_type=cfg[type]['peft_type'],
            task_type=cfg[type]['task_type'],
            r=int(cfg[type]['r']),
            lora_alpha=int(cfg[type]['lora_alpha']),
            lora_dropout=float(cfg[type]['lora_dropout']),
            target_modules=list(cfg[type]['target_modules']),
        )
       lora_model = get_peft_model(model = model, peft_config = peft_args)
       
       return lora_model
    
    elif type == 'adalora':
        peft_args=AdaLoraConfig(
            peft_type=cfg[type]['peft_type'],
            task_type=cfg[type]['task_type'],
            init_r=int(cfg[type]['init_r']),
            lora_alpha=int(cfg[type]['lora_alpha']),
            lora_dropout=float(cfg[type]['lora_dropout']),
            target_modules=list(cfg[type]['target_modules']),
            total_step=int((cfg['training']['num_train_epochs']*cfg['training']['num_generations']*sample_size)/cfg['training']['per_device_train_batch_size'])
        )
        #sanity-check
        print(f'Total train steps in adalora: {peft_args.total_step}')
        adalora_model = AdaLoraModel(model = model, 
                                     config = peft_args, 
                                     adapter_name = 'Default') 

        return adalora_model


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
        #logging and saving config
        logging_strategy=cfg['training']['logging_strategy'],
        logging_steps=float(cfg['training']['logging_steps']),
        logging_dir=cfg['training']['logging_dir'],
        save_strategy=cfg['training']['save_strategy'],
        save_steps=float(cfg['training']['save_steps']),
        temperature=cfg['generation']['temperature'],
        max_prompt_length=max_prompt_length,
        gradient_checkpointing=True, #memory debug
        report_to=cfg['training']['report_to'] #keeping it w 
        #try for memory debug: disable_dropout=true
        #try for length based think trace: loss_type="dr_grpo"
    )
    return training_args

def train(train_data,peft_type=None):

    #sanity-checks
    print(f"Train data size is:{len(train_data)}")
    #fetch max prompt length param for this data
    max_prompt_length=max([len(tokenized_prompt) for tokenized_prompt in train_data['input_ids']])
    print(f'MAX PROMPT LENGTH: {max_prompt_length}')

    training_args=get_train_args(max_prompt_length)
    model.generation_config.temperature = training_args.temperature
    model.generation_config.min_new_tokens = 600
    model.generation_config.max_new_tokens = 1024
    model.get_input_embeddings().weight.requires_grad = True
    model.enable_input_require_grads()

    trainer = GRPOTrainer(
        model = model,
        reward_funcs=[
            format_reward,
            plan_reward,
        ],
        args = training_args,
        train_dataset = train_data,
        processing_class = tokenizer,
        peft_config = peft_args
    )

    #sanity-check    
    print(f"🔍 Model after GRPOTrainer wrapping:{type(trainer.model)}")

    #check current generation temperature
    print(f"MODEL GENERATION CONFIG: {trainer.model.generation_config.to_dict()}")

    # print(f"DEBUG model params in gradient required mode:")
    # for name, param in trainer.model.named_parameters():
    #     if param.requires_grad:
    #         print(name)
    trainer.train()

    # print("\n🔍 GPU Memory Summary after training:\n")
    # print(torch.cuda.memory_summary())
    return trainer

def main(n,split,sample_size,peft_type=None):
    
    #fetch train and eval data
    if n == 4:
        three_train = GRPO_utils.GRPO_load_tokenized_data(3)
        four_train = GRPO_utils.GRPO_load_tokenized_data(4)
    else:
        train_data=GRPO_utils.GRPO_load_tokenized_data(n,split)

        if sample_size is not None:
            train_data=train_data.select(range(sample_size))

    # #combine three and four block dataset to one
    train_data=concatenate_datasets([three_train,four_train])

    train(train_data=train_data,peft_type=peft_type)
    
    num_generations=int(cfg['training']['num_generations'])
    epoch_len=num_generations*len(train_data)

    eval_df,problems_df=eval.epoch_eval(tracker.format_scores,tracker.plan_scores,tracker.terminate,epoch_len=epoch_len,num_generations=num_generations)
    path=cfg['training']['output_dir']
    eval_df.to_csv(path+'/train_metrics.csv')
    problems_df.to_csv(path+'/problem_metrics.csv')
    print(f"Metrics saved to : {path}")

main(n=4, split='train', sample_size=None)