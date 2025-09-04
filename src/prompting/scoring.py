import sys
sys.path.append("/srv/chawak/planning-with-llms/src/")

import regex as re
import torch

from prompting import scoring_utils
from prompting import scoring_eval


#----------------------------- reward functions ----------------------------- 
#<think>(.*?)<\/think>
format_pattern=r"\[PLAN\](.*?)\[PLAN END\]"
format_pattern=re.compile(format_pattern, re.DOTALL)

#expects model response, returns a float score
def compute_format_reward(response, **kwargs):

    print(f'In format reward, model response is: {repr(response)}')

    score=0
    #hard format reward
    if format_pattern.fullmatch(response.strip()):
        score = 10
    #soft format reward
    elif format_pattern.search(response.strip()):
        score = 2
    print(f"Format reward for this response is: {score}")

    return score

#expects model response, returns a float score
def compute_plan_reward(response,init, goal, gold_plan, **kwargs):
    
    #response score returns a tuple of plan and bonus scores
    scores=scoring_utils.response_score(response=response, init=init, goal=goal, gold_plan=gold_plan)           

    return scores

#----------------------------- main set-up -----------------------------

#expects a list of completions, init, goal and gold plan and
#outputs the mean scores across the generated plans for the sample dataset size

def score(completions,init_list,goal_list,gold_plan_list,sample_size):

    format_reward=[]
    plan_reward=[]
    bonus_reward=[]
    
    for idx,completion in enumerate(completions):
        #score each completion
        f = compute_format_reward(completion)
        scores = compute_plan_reward(response=completion,init=init_list[idx],goal=goal_list[idx],gold_plan=gold_plan_list[idx])
        p = scores[0]
        b = scores[1]    
        format_reward.append(f)
        plan_reward.append(p)
        bonus_reward.append(b)
    eval_df=scoring_eval.eval(
        format_scores=format_reward,
        plan_scores=plan_reward,
        bonus_scores=bonus_reward,
        sample_size=sample_size)
    
    path='/srv/chawak/planning-with-llms/results/Vanilla_Prompting_new'
    eval_df.to_csv(path+'/0107_no4bit_score_metrics.csv')



