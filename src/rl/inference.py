import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from unsloth import FastLanguageModel

import sys
sys.path.append("/home/user/planning-with-llms/src")

import torch
import shared.policy_model as policy
from huggingface_hub import login

from shared import unifiedplanning_blocksworld as bw, planbench as pb, prompts

import pandas as pd
import regex as re
import math
import time

cpt=120 #cpt
login(token="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")
base_dir='/home/user/planning-with-llms/results/rl/training/'


def get_Unsloth_model():

    # Load the model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="google/gemma-3-12b-it",
        max_seq_length=1024,
        device_map='auto',
        load_in_4bit=False,  # False for LoRA 16bit
        fast_inference=False,
        gpu_memory_utilization=0.8,
    )
    return model,tokenizer


def infer(prompt,model,temp=0):
    #query local model and fetch decoded response
    pi=policy.PolicyModel()
    print(f"Model adapter received by infer function : {getattr(model, 'adapter_path', 'Unknown')}")
    response=pi.GRPO_one_shot(prompt,model,temp)
    return response

def apply_and_get_model():
    #time logs
    start_time=time.time()
    model,tokenizer=get_Unsloth_model()

    #base_model,tokenizer=get_Unsloth_model()
    #model=base_model #init model is base

    model_path=base_dir+f'debug-2006/checkpoint-{cpt}' #cpt
    model.load_adapter(model_path)

    #peft_model=PeftModel.from_pretrained(model,model_path)#,is_trainable=False,adapter_name="default")
    #peft_model.adapter_path=model_path
    print(f'\n\n++++++++ Loading model from: {model_path}')

    #saving GPU memory 
    torch.cuda.empty_cache()
    model.generation_config.use_cache = False 
    model.generation_config.temperature = 0.1
    model.config.use_cache = False
    #sanity-check
    print("Generation config:", model.generation_config.to_dict())
    FastLanguageModel.for_inference(model)

    #time logs
    end_time=time.time()
    print(f'Model loading and merging took {end_time-start_time:.2f} seconds.') 
    return model

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

def main(df):

    model_it=0
    peft_model=None
    
    for model_it in range(0,1):

        #default-initialisations
        results=[False]*len(df)
        well_struct=[False]*len(df)
        num_tries=[0]*len(df)
        diff_planlen=[-math.inf]*len(df)
        actions_to_goal=[-math.inf]*len(df)
        gplan_lens=[-math.inf]*len(df)
        valid_action_count=[0]*len(df)
        
        #load pre-trained PEFT model for iteration it 
        peft_model=apply_and_get_model()
        
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
            prompt=prompt[:-9]
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
            model_plan=problem.GRPOcreate_plan_from_tuples(action_tuples=actions)
            if not model_plan:
                results[i]=False
                print('!!!!!!!!!!!Action statement NOT FROM LIST OF ALLOWED actions!!!!!!!!!!!!')

            if model_plan:
                print(f'LLM PLAN: {model_plan}')


            def apply_plan(problem,model_plan):
                
                #validate and apply    
                simulation=problem.create_seq_simulation()
                va_counter=0
                valid_state,va_counter,distance2goal=problem.GRPO_check_and_apply(simulation,model_plan)
                #check distance to goal for last valid state
                d=problem.actions_to_goal(valid_state)
                distance2goal.append(d)
                
                return valid_state,va_counter,distance2goal



            #validate and apply    
            simulation=problem.create_seq_simulation()
            va_counter=0
            valid_state,va_counter,distance2goal,flag=problem.GRPO_check_and_apply(simulation,model_plan)
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
            gplan_actions=gplan_actions[1:-1]
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
        path=f'{base_dir}/inference/inference_24_06/{split}/tags_checkpoint_{cpt}' #cpt
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

n=3
split='train'
data_dir=f"/home/user/planning-with-llms/data/{n}_blocks"
data_path=f'{data_dir}/SFT_{split}_{n}_blocks_fullPlan'
eval_three=pd.read_csv(data_path)
eval_three=eval_three.drop(columns=['Unnamed: 0'])
eval_data=eval_three

print(f"Eval data length: {len(eval_data)}")
main(df=eval_data)