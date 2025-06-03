import torch
from peft import PeftModel, PeftConfig
from huggingface_hub import login
import pandas as pd
import regex as re
import math
from copy import deepcopy
from peft.tuners.lora import LoraLayer
import time
import os
import sys

sys.path.append("/srv/chawak/planning-with-llms/src")
import shared.policy_model as policy
import shared.llm_utils as llm_utils
import shared.unifiedplanning_blocksworld as bw
import shared.planbench as pb
import shared.prompts as prompts

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
login(token="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")
base_dir='/srv/chawak/planning-with-llms/results/SFT'

cpts=[0,15,30,45,60,75,90,105,120,135,140]

def infer(prompt,model,temp=0):
    #query local model and fetch decoded response
    pi=policy.PolicyModel()
    response=pi.SFT_one_shot(prompt,model,temp)
    return response

def apply_and_get_model(it):
    #time logs
    start_time=time.time()
    base_model,tokenizer=llm_utils.get_model_tokenizer()

    if it==0:
        return base_model

    else:
        model=base_model #init model is base
        model_path=base_dir+f'/training/training_16-05/checkpoint-{cpts[it]}'
       
        peft_model=PeftModel.from_pretrained(model,model_path,is_trainable=False,adapter_name="default")
        print(f'\n\n++++++++ Loading model from: {model_path}')

        #saving GPU memory 
        torch.cuda.empty_cache()
        peft_model.generation_config.use_cache = False 
        peft_model.config.use_cache = False
        
        #time logs
        end_time=time.time()
        print(f'Model loading and merging took {end_time-start_time:.2f} seconds.') 
        return peft_model

def evaluate(attempts,struct,diff_len,results,actions_to_goal,gplan_lens,valid_action_count):
    metrics=pd.DataFrame({
        'Parsable':struct,
        'Attempts':attempts,
        'Terminated':results,
        'Valid Actions':valid_action_count,
        'Distance from goal':actions_to_goal,
        'Diff in plan len':diff_len,
        'GOLD Plan len':gplan_lens
    })
    correct_plans=sum(results)
    c_rate=correct_plans/len(results)
    c_rate=format(c_rate,".2f")
    return metrics,c_rate


def main(df,iterations):

    # metric_list=[pd.DataFrame() for _ in range(iterations)]
    # c_rate=[0]*iterations
    model_it=0
    peft_model=None
    
    for model_it in range(10,11):

        #default-initialisations
        results=[False]*len(df)
        well_struct=[False]*len(df)
        num_tries=[0]*len(df)
        diff_planlen=[-math.inf]*len(df)
        actions_to_goal=[-math.inf]*len(df)
        gplan_lens=[-math.inf]*len(df)
        valid_action_count=[0]*len(df)
        
        #load pre-trained PEFT model for iteration it 
        peft_model=apply_and_get_model(model_it)
        
        #time logs
        start_time=time.time()

        for i in range(0,len(df)):
        #for i in range(0,1):
            #print(f'Loaded model: {peft_model}')    
            if peft_model is None:
                print(f"Warning: Model for iteration {model_it} could not be loaded. Skipping.")
                continue
        

            print(f'\n\n--------------------Entering main loop PROBLEM #{i}--------------------')
            
            #declarations: Blocksworld Problem object
            problem = bw.BlocksworldProblem()
            model_plan=None

            #extract init,goal,prompt from dataset
            init=prompts.parse_init(df.iloc[i]['init'])
            goal=prompts.parse_goal(df.iloc[i]['goal'])
            prompt=df.iloc[i]['prompt']
            
            
            pb.parse_planbench_initial_condition(problem, init)
            pb.parse_planbench_goal_state(problem, goal)
            print(f'\n\nBlocksworld Problem Initial Values:{problem.initial_values}')
            print(f'\nBlocksworld Problem Goal State:{problem.goals}')

            #activate solve mode
            tags="Answer within the [PLAN] [PLAN END] tags."
            response=infer(prompt=prompt+'\n'+tags,model=peft_model,temp=0.1)
            actions=response[0]
            tries=response[1]

            num_tries[i]=tries

            if not actions:
                well_struct[i]=False
                print(f'\n\n LLM exceeded 3 tries')
                results[i]=False
                continue
            well_struct[i]=True
            plan_len=len(actions)

            print(f'\n\nProposed actions by LLM : {actions} \n  END')
            model_plan=problem.create_plan_from_tuples(actions)
            if not model_plan:
                results[i]=False
                print('!!!!!!!!!!!Action statement NOT FROM LIST OF ALLOWED actions!!!!!!!!!!!!')

            if model_plan:
                print(f'LLM PLAN: {model_plan}')

            #validate and apply    
            simulation=problem.create_seq_simulation()
            va_counter=0
            valid_state,flag,va_counter=problem.check_and_apply(simulation,model_plan)
            valid_action_count[i]=va_counter
            if not flag:
                results[i]=False

            #compute action-steps away from goal for last valid-state
            # print(f'UBS OBJECT FOR LAST VALID STATE: {valid_state,sim}')
            a2g=problem.actions_to_goal(valid_state)
            actions_to_goal[i]=a2g
            print(f'Actions to goal count is:{a2g}')
            
            #get gold-plan on problem
            gold_plan=df.iloc[i]['gold_plan']
            gplan_actions=str(gold_plan).strip().split('\n')
            gplan_actions=gplan_actions[1:]
            gplan_len=len(gplan_actions)
            gplan_lens[i]=gplan_len
            print(f'\n\n$$$Proposed actions by GOLD PLAN $$$ : {gplan_actions} \n END')


            #termination-check 
            if problem.terminate(valid_state):
                results[i]=True
                diff_planlen[i]=plan_len-gplan_len
                continue
            
            results[i]=False


        #compute metrics
        metrics,c_rate=evaluate(num_tries,well_struct,diff_planlen,results,actions_to_goal,gplan_lens,valid_action_count)
        print(f"\n--- Metrics for model-iteration {model_it} ---")
        print(metrics)
        path=f'{base_dir}/inference/inference_16-05/{split}/epoch_{model_it}'
        os.makedirs(path, exist_ok=True) 
        metrics.to_csv(os.path.join(path, "metrics.csv"))
        print(f'Metrics saved to {path}')
        #time logs
        end_time=time.time()
        print(f'Evaluation on model iteration {model_it} took {end_time-start_time:.2f} seconds.') 

        #saving GPU memory 
        torch.cuda.empty_cache()
        del peft_model
    return



#load evaluation dataset
n=3
split='val'
data_dir=f"/srv/chawak/planning-with-llms/data/{n}_blocks"
data_path=f'{data_dir}/SFT_{split}_{n}_blocks_fullPlan'
eval_data=pd.read_csv(data_path)
eval_data=eval_data.drop(columns=['Unnamed: 0'])

iterations=10
main(df=eval_data,iterations=iterations)


#debug
# lora_layers=[module for module in peft_model.modules() if isinstance(module, LoraLayer)]
# print(f'Number of lora layers in peft model : {len(lora_layers)}')

# total_params = sum(p.numel() for p in peft_model.parameters())
# trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
# print(f"Total parameters: {total_params}")
# print(f"Trainable parameters: {trainable_params}")

# for name, param in peft_model.named_parameters():
#     if "lora_" in name:
#         print(f"{name} → mean: {param.data.mean().item():.7f}, std: {param.data.std().item():.7f}")

#debug 
# from safetensors.torch import load_file
# path = base_dir+f"iteration_{model_it}/adapter_model.safetensors"
# state_dict = load_file(path)
# print("##############PEFT model keys ##############")
# for k in state_dict.keys():
#     print(k)