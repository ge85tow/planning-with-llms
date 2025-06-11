from unsloth import FastLanguageModel
import sys
sys.path.append("/srv/chawak/planning-with-llms/src")

from shared import llm_utils
from shared import unifiedplanning_blocksworld as bw
from shared import prompts
from shared import planbench as pb
import regex as re

def make_action_tuples(response):

    #get action tuples
    action_tuples=llm_utils.parse_action_tuples(response)
    if not action_tuples:
        print(f'\n\n Imparsable response')
        return False
    
    return action_tuples

def response2plan(problem,init,goal,response,):
    
    model_plan=None
    #create a blocksworld problem 
    init=prompts.parse_init(init)
    goal=prompts.parse_goal(goal)
    pb.parse_planbench_initial_condition(problem, init)
    pb.parse_planbench_goal_state(problem, goal)
    print(f'\n\nBlocksworld Problem Initial Values:{problem.initial_values}')
    print(f'\nBlocksworld Problem Goal State:{problem.goals}')

    #get model plan
    action_tuples=make_action_tuples(response)
    if action_tuples:
        model_plan=problem.GRPOcreate_plan_from_tuples(action_tuples)
    
    print(f"Model responded with this plan: {model_plan}")
    return action_tuples,model_plan

def apply_plan(problem,model_plan):
    
    #validate and apply    
    simulation=problem.create_seq_simulation()
    va_counter=0
    valid_state,va_counter,distance2goal=problem.GRPO_check_and_apply(simulation,model_plan)
    #check distance to goal for last valid state
    d=problem.actions_to_goal(valid_state)
    distance2goal.append(d)
    
    return valid_state,va_counter,distance2goal

def goal_proximity(distance2goal:list) -> list:
    
    scores=[]
    
    for idx,d in enumerate(distance2goal):
        
        if d == 0: #goal state reached
            break

        if(idx<=len(distance2goal)-2):
            # print("-"*20,f"IDX:{idx}")
            d_old=d
            d_new=distance2goal[idx+1]
            score=max(0,(d_old-d_new)*5)
            scores.append(score)
 
    return scores

def get_plan_len(plan):    
    
    plan_len=len(plan.split('\n'))-2

    return plan_len

#scores gold plan
def gold_plan_reward(problem, gold_plan):
    
    print('~'*30,'\n ATTENTION: THIS IS ALL FOR GOLD PLAN REWARD')
    score=0

    #get gold-plan plan object
    action_tuples=make_action_tuples(gold_plan)
    gold_plan_ob=problem.GRPOcreate_plan_from_tuples(action_tuples)
    current_state,va_counter,distance2goal= apply_plan(problem,gold_plan_ob)

    #score + 2 for each valid action
    score+= va_counter*2
    #score for when we are moving towards goal state
    # print(f"GOLD-PLAN distance metric is: {distance2goal}")
    distance_scores=goal_proximity(distance2goal)
    # print(f"GOLD-PLAN proximity scores are: {distance_scores}")
    score+=sum(distance_scores)
    # print(f"GOLD-PLAN scores without bonus reward{score}")

    print('GOLD PLAN SCORE IS: ',score)
    print('~'*30, '\n END ATTENTION: GOLD PLAN REWARD COMPUTE END')
    return score

#bonus rewards: correct termination, optimality
def bonus_reward(problem,valid_state,plan_len,goldplanlen):
    
    score=0
    #check if plan terminates to goal
    if problem.terminate(valid_state):
        score+=20 #20 for reaching goal
        
        #check if model plan matches the gold plan length
        if plan_len==goldplanlen: score+=10

        #check if model plan superceeds the gold plan length
        if plan_len<goldplanlen: score+=15

    return score

#scores plan correctness
def response_score(response,init,goal,gold_plan):

    print('-'*20,"Entering response score",'-'*20)
    print(f'\n Model Response: {response}')
    #define blocksworld problem 
    problem=bw.BlocksworldProblem()
    score = 0

    #VALID ACTION REWARD:
    #extract valid actions from response
    action_tuples,model_plan=response2plan(problem=problem,init=init,goal=goal,response=response)
    
    #if there was a non-empty plan generated
    if model_plan:
        current_state,valid_action_count,distance2goal=apply_plan(problem=problem,model_plan=model_plan)    
        print(f'Distance 2 goal metric:{distance2goal}')
        gold_plan_len=get_plan_len(gold_plan)
        #limited to total number of actions possible, to avoid reward hacking
        score+=min(gold_plan_len*2,valid_action_count*2)
        print(f"Valid actions score is: {score}")

        #PROXIMITY REWARD:
        distance_scores=goal_proximity(distance2goal)
        print(f"Proximity scores are: {distance_scores}")
        score+=sum(distance_scores)

        print(f"Score without bonus reward: {score}")

        #normalize model-plan's reward by the gold plan reward to 0-60 range
        gold_plan_score=gold_plan_reward(problem=problem,gold_plan=gold_plan)
        score=(score/gold_plan_score)*60

        #bonus rewards for correct termination and optimality
        score+=bonus_reward(problem,current_state,len(action_tuples),gold_plan_len)
        print(f"Score with bonus reward: {score}")

    print('!'*80,f"FINAL SCORE FOR THIS GENERATION: {score} \n\n\n")
    return score

def responses_scores(response_list,init_list,goal_list,gold_plan_list):
    scores=[]
    for idx in range(len(response_list)):
        score = response_score(response=response_list[idx],init=init_list[idx],goal=goal_list[idx],gold_plan=gold_plan_list[idx])
        scores.append(score)
    return scores

#-----------------------------------llm utils------------------------------------------------------
import torch
from datasets import load_from_disk
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'

lora_rank=16
cache_dir='/home/chawak/huggingface'

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="google/gemma-3-12b-it",
    fast_inference=True,  # Enable vLLM fast inference
    max_lora_rank=lora_rank,
    device_map='auto',
    cache_dir=cache_dir,
    # torch_dtype=torch.bfloat16,
    attn_implementation='eager',
    gpu_memory_utilization=0.85  # Reduce if out of memory
)

peft_model = FastLanguageModel.get_peft_model(
    model,
    r=lora_rank,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj"],  # Remove QKVO if out of memory
    lora_alpha=lora_rank,
    use_gradient_checkpointing="unsloth",  # Enable long context finetuning
    task_type= "CAUSAL_LM",
    random_state=3407)

def GRPO_load_tokenized_data(n):
    data_dir=f"/srv/chawak/planning-with-llms/data/{n}_blocks"
    #load train dataset
    split='train'
    data_path=f'{data_dir}/GRPO_tokenized_dataset/{split}'

    #change for toy example
    #data_path=f'/srv/chawak/planning-with-llms/data/toy_label_nopad/{split}'

    train_data=load_from_disk(data_path)

    #debug 
    #train_data=train_data.select(range(1))
    # print(f'The train data is: {train_data}')
    # print(f'------------------Train data------------------')
    # print(f"Input ids tokens: {train_data['input_ids'][0]}")
    # print(f"Length of input ids tokens: {len(train_data['input_ids'][0])}")
    # print(f"Label tokens: {train_data['labels'][0]}")
    # print(f"Length of label tokens: {len(train_data['labels'][0])}")   
    return train_data  
        

