import pandas as pd
from statistics import mean
# import sys
# sys.path.append("/home/user/planning-with-llms/src/")

def get_epoch_metric(metric_list,epoch_len):
    
    metric_chunks=[]
    #split metrics list by step-size
    for i in range(0, len(metric_list), epoch_len):
        metric_chunk=metric_list[i:i+epoch_len]
        metric_chunks.append(metric_chunk)

    return metric_chunks


def compute_epoch_mean(metric_chunks):
    mean_values=[mean(metric) for metric in metric_chunks]
    return mean_values


def goal_reached(bonus_per_epoch):
    
    terminate=[]
    for bonus_score in bonus_per_epoch:
        terminate_per_epoch=[]
        for score in bonus_score:
            terminate_per_epoch.append(1 if score>0 else 0) #count problems where plan reaches goal state
        terminate.append(terminate_per_epoch)

    return terminate

def epoch_eval(format_scores,plan_scores,bonus_scores,epoch_len):

    format_per_epoch=get_epoch_metric(metric_list=format_scores,epoch_len=epoch_len)
    plan_per_epoch=get_epoch_metric(metric_list=plan_scores,epoch_len=epoch_len)
    bonus_per_epoch=get_epoch_metric(metric_list=bonus_scores,epoch_len=epoch_len)

    mean_epoch_format=compute_epoch_mean(format_per_epoch)
    mean_epoch_plan=compute_epoch_mean(plan_per_epoch)
    mean_epoch_bonus=compute_epoch_mean(bonus_per_epoch)

    terminate=goal_reached(bonus_per_epoch)

    eval_df=pd.DataFrame({
                        'reward/format_reward': mean_epoch_format,
                        'reward/plan_reward': mean_epoch_plan,
                        'reward/bonus_reward': mean_epoch_bonus,
                        '# Terminate': [sum(t) for t in terminate],
                        'Terminate': terminate
                        })
    print(f"Epoch-wise rewards are {eval_df}")

    

    return eval_df





    