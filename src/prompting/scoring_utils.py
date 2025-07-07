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
    valid_state,va_counter,distance2goal,flag=problem.GRPO_check_and_apply(simulation,model_plan)
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
    
    print('~'*70,'\n ATTENTION: THIS IS ALL FOR GOLD PLAN REWARD')
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
    print('~'*70, '\n END ATTENTION: GOLD PLAN REWARD COMPUTE END')
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

#parses the plan for correctness in blocksworld
def response_score(response,init,goal,gold_plan):

    # print('-'*70,f"Entering response score for completion {scoring_tracker.total_completions_seen}",'-'*70)
    print(f'\n Model Response: {response}')
    #define blocksworld problem 
    problem=bw.BlocksworldProblem()
    score = 0
    bonus_score=0

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
        score=round((score/gold_plan_score)*60, 2)

        #bonus reward scores: correct termination and optimality
        bonus_score=bonus_reward(problem,current_state,len(action_tuples),gold_plan_len)
        print(f"Score with bonus reward: {score+bonus_score}")
        
    # scoring_tracker.total_completions_seen+=1
    print('!'*80,f"FINAL SCORE FOR THIS GENERATION: {score+bonus_score} \n\n\n")
    
    return score,bonus_score
