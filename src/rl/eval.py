import pandas as pd
from statistics import mean
import sys
sys.path.append("/home/chawak/vast-ai/planning-with-llms/src/")
from rl import tracker
from collections import defaultdict
problem_list = defaultdict(list)

def get_epoch_metric(metric_list,epoch_len):
    
    metric_chunks=[]
    #split metrics list by step-size
    for i in range(0, len(metric_list), epoch_len):
        metric_chunk=metric_list[i:i+epoch_len]
        metric_chunks.append(metric_chunk)

    return metric_chunks

#expects: list of metric list, returns: average value 
def compute_epoch_mean(metric_chunks):
    mean_values=[mean(metric) for metric in metric_chunks]
    return mean_values

#expects: bonus score metric for each epoch, 
#returns: number of terminating plans 
def goal_reached(bonus_per_epoch):
    
    terminate=[]
    for bonus_score in bonus_per_epoch:
        terminate_per_epoch=[]
        for score in bonus_score:
            terminate_per_epoch.append(1 if score>0 else 0) #count problems where plan reaches goal state
        terminate.append(terminate_per_epoch)

    return terminate

def compute_num_terminate(terminate_per_epoch_per_problem):
    num_terminate = 0
    for element in terminate_per_epoch_per_problem:
        #print(f"ELEMENT:{element}")
        if element[0]>0:
            num_terminate+=1
    return num_terminate

def track_problem_termination(terminate_per_epoch,num_generations):

    #iterate through bonus scores for each epoch 
    for idx,epoch_bonus_scores in enumerate(terminate_per_epoch):
        
        # print(f"BONUS SCORE FOR EPOCH {idx}: {epoch_bonus_scores}")

        terminate_per_epoch_per_problem=get_epoch_metric(metric_list=epoch_bonus_scores,epoch_len=num_generations)

        # print(f"Terminate per epoch per problem: {terminate_per_epoch_per_problem}")

        # terminate_per_epoch_per_problem=terminate_per_epoch_per_problem[0]
        for problem_scores in terminate_per_epoch_per_problem:
            # print(f"Problem scores: {problem_scores}")

            num_terminate=compute_num_terminate(problem_scores)
            state_config=problem_scores[0][1]
            # print(f"state config: {state_config}")

            if state_config not in problem_list:
                problem_list[state_config]=[]

            problem_list[state_config].append(num_terminate)

            # print(f"Problem list state config in track problem:{problem_list[state_config]}")

    return problem_list

def epoch_eval(format_scores,plan_scores,terminate,epoch_len,num_generations):

    format_per_epoch=get_epoch_metric(metric_list=format_scores,epoch_len=epoch_len)
    plan_per_epoch=get_epoch_metric(metric_list=plan_scores,epoch_len=epoch_len)
    mean_epoch_format=compute_epoch_mean(format_per_epoch)
    mean_epoch_plan=compute_epoch_mean(plan_per_epoch)

    #get bonus scores from terminate tuples
    bonus_scores=[]
    for t in terminate:
        bonus_score=t[0]
        bonus_scores.append(bonus_score)
    bonus_per_epoch=get_epoch_metric(metric_list=bonus_scores, epoch_len=epoch_len)
    mean_epoch_bonus=compute_epoch_mean(bonus_per_epoch)

    #sanity check
    # print(f"TERMINATE BEFORE SPLITTING: {terminate}")

    terminate_per_epoch=get_epoch_metric(metric_list=terminate,epoch_len=epoch_len)
    
    epoch_track_problems=track_problem_termination(terminate_per_epoch=terminate_per_epoch, num_generations=num_generations)
    # print(f'Epoch wise problem tracking: {epoch_track_problems}')
    eval_df=pd.DataFrame({
                        'reward/format_reward': mean_epoch_format,
                        'reward/plan_reward': mean_epoch_plan,
                        'reward/bonus_reward': mean_epoch_bonus#,
                        # '# Terminate': [sum(t) for t in terminate],
                        #'Terminate': terminate
                        })
    problem_df = pd.DataFrame.from_dict(epoch_track_problems, orient='index')
    # print(f"BIGGEST SANITY CHECK EVER:{epoch_track_problems[0][1]}")
    # problem_df.columns=list(range(len(epoch_track_problems[0][1])))
    print(f"Epoch-wise problem tracking: {problem_df}")
    print(f"Epoch-wise rewards are {eval_df}")

    

    return eval_df,problem_df





# for outer_idx, bonus_score in enumerate(bonus_per_epoch):
# for inner_idx,bonus in enumerate(bonus_score):
#     problem_idx=(outer_idx+1)*inner_idx
#     key=terminate[problem_idx][1]
#     print(f"COMPUTED KEY FOR PROBLEM LIST: {problem_list[key]}")
#     problem_score=compute_problem_scores()

# t=compute_num_terminate([(20,('a','b')),
#                        (10,('a','b')),
#                        (0,('a','b')),
#                        (0,('a','b'))
# ])
# print(f'T is : {t}')