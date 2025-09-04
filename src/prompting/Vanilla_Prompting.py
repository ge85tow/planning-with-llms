import sys
sys.path.append("/srv/chawak/planning-with-llms/src")
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from shared import unifiedplanning_blocksworld as bw
from shared import prompts
from shared import planbench as pb
from shared import policy_model as policy
from prompting import completions_tracker
from prompting import scoring

import pandas as pd
import regex as re
import math

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

def main(df,model,temp):
    results=[False]*len(df)
    well_struct=[False]*len(df)
    num_tries=[0]*len(df)
    diff_planlen=[-math.inf]*len(df)
    actions_to_goal=[-math.inf]*len(df)
    gplan_lens=[-math.inf]*len(df)
    valid_action_count=[0]*len(df)
    #activate solve mode
    pi = policy.PolicyModel()
    
    for i in range(0,len(df)):
    # for i in range(0,2):
        
        print(f'\n\n--------------------Entering main loop PROBLEM #{i}--------------------')
        
        #declarations: Blocksworld Problem object
        problem = bw.BlocksworldProblem()
        model_plan=None

        #extract init,goal,prompt from dataset
        init=prompts.parse_init(df.iloc[i]['init'])
        goal=prompts.parse_goal(df.iloc[i]['goal'])
        prompt=df.iloc[i]['prompt']
            
        #changed generation of Blocksworld object for OUR dataset
        pb.parse_planbench_initial_condition(problem, init)
        pb.parse_planbench_goal_state(problem, goal)
        print(f'\n\nBlocksworld Problem Initial Values:{problem.initial_values}')
        print(f'\nBlocksworld Problem Goal Condition:{problem.goals}')

        #query the llm, parse the actions, return tuples
        # tags="Answer within the [PLAN] [PLAN END] tags."
        think="I first think about the reasoning process in the mind and then provide the user with the plan."
        think+=" The reasoning process and plan are enclosed within <think> </think> and [PLAN] [PLAN END] tags, respectively,"
        think+=" i.e., <think> reasoning process here </think> [PLAN] plan here [PLAN END]."

        response=pi.Vanilla_one_shot(prompt=prompt+"\n"+think,temp=temp)
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
        print(f"LAST VALID STATE after check & apply ends: {valid_state}")
        valid_action_count[i]=va_counter
        if not flag:
            results[i]=False

        #compute action-steps away from goal for last valid-state
        # print(f'UBS OBJECT FOR LAST VALID STATE: {valid_state,sim}')
        a2g=problem.actions_to_goal(valid_state)
        actions_to_goal[i]=a2g

        
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
    return metrics,c_rate

n=3
model='google/gemma-3-12b-it'
temp=0.1
split='val'

data_dir=f"/srv/chawak/planning-with-llms/data"
df=pd.read_csv(f'{data_dir}/{n}_blocks/SFT_{split}_{n}_blocks_fullPlan')
df.drop(columns='Unnamed: 0',inplace=True)

response=main(df,model=model,temp=temp)
metrics_df=response[0]
c_rate=response[1]

print(f'----------------model={model},temp={temp}, on dataset split={split} w plan-end tags----------')
print(metrics_df.to_markdown())
results_dir="/srv/chawak/planning-with-llms/results/Vanilla_Prompting_new"
metrics_df.to_csv(f'{results_dir}/Oneshot_{split}_{n}_blocks')

scoring.score(completions=completions_tracker.completions,
    init_list=list(df['init']),
    goal_list=list(df['goal']),
    gold_plan_list=list(df['gold_plan']),
    sample_size=len(df))
