from shared import unifiedplanning_blocksworld as bw
from shared import prompts
from shared import planbench as pb
from shared import policy_model as policy
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


def main(df,thoughts,model,temp):
    results=[False]*len(df)
    well_struct=[False]*len(df)
    num_tries=[0]*len(df)
    diff_planlen=[-math.inf]*len(df)
    actions_to_goal=[-math.inf]*len(df)
    gplan_lens=[-math.inf]*len(df)
    valid_action_count=[0]*len(df)
    for i in range(0,len(df)):
    #for i in range(0,1):
        
        print(f'\n\n--------------------Entering main loop PROBLEM #{i}--------------------')
        
        #declarations: Blocksworld Problem object
        problem = bw.BlocksworldProblem()
        model_plan=None
        cot="Strategise at each step before you choose an action, then form a final plan."
        #create prompt using OUR dataset
        prompt,init,goal=prompts.cot_make_prompt(df.iloc[i]['init'],
                                                 df.iloc[i]['goal'],
                                                 df.iloc[i]['demo_init'],
                                                 df.iloc[i]['demo_goal'],
                                                 df.iloc[i]['demo_plan'],
                                                 reason=thoughts.iloc[i]['thought'],
                                                 incantation=cot)
        
        #changed generation of Blocksworld object for OUR dataset
        pb.parse_planbench_initial_condition(problem, init)
        pb.parse_planbench_goal_state(problem, goal)
        print(f'\n\nBlocksworld Problem Initial Values:{problem.initial_values}')
        print(f'\nBlocksworld Problem Goal Condition:{problem.goals}')

        #activate solve mode
        pi = policy.PolicyModel()
        #query the llm, parse the actions, return tuples
        #tags="Answer within the [PLAN] [PLAN END] tags."
        #cot="Strategise at each step before you choose an action, then form a final plan."
        response=pi.Vanilla_fullSol_one_shot(prompt,model=model,temp=temp)
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

        
        #get gold-plan on problem
        gold_plan=problem.generate_plan()
        gplan_actions=str(gold_plan.plan).strip().split('\n')
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
    return metrics,c_rate

n=3
model='google/gemma-2-9b-it'
temp=0
split='train'

#read data
df=pd.read_csv(f'../data/{n}_blocks/Oneshot_{split}_{n}_blocks')
#df=pd.read_csv(f'../data/{n}_blocks/Oneshot_{split}_{n}_blocks_merged') #!!!!!!!!!!!!! only for n=5
#df=pd.read_csv(f'../data/{n}_blocks/contaminated_test_{n}_blocks')
df.drop(columns='Unnamed: 0',inplace=True)
thoughts=pd.read_csv(f'../data/{n}_blocks/{split}_thoughts.csv').drop(columns='Unnamed: 0')
#thoughts=pd.read_csv(f'../data/{n}_blocks/contaminated_test_thoughts.csv').drop(columns='Unnamed: 0')

#parse response and metrics
response=main(df,model=model,temp=temp,thoughts=thoughts)
metrics_df=response[0]
c_rate=response[1]

print(f'----------------model={model},temp={temp}, on dataset split={split} w plan-end tags----------')
print(metrics_df.to_markdown())
metrics_df.to_csv(f'../results/DeepThink_COT_{split}_{n}_blocks')
#metrics_df.to_csv(f'../results/DeepThink_COT_contaminated_test_{n}_blocks')