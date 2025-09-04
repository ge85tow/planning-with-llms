import pandas as pd
from statistics import mean


def goal_reached(bonus_scores):
    
    terminate=[]
    for score in bonus_scores:
        terminate.append(1 if score>0 else 0) #count problems where plan reaches goal state

    return terminate 

def eval(format_scores,plan_scores,bonus_scores,sample_size):

    mean_format=mean(format_scores)
    mean_plan=mean(plan_scores)
    mean_bonus=mean(bonus_scores)
    terminate=goal_reached(bonus_scores)

    eval_df=pd.DataFrame({
                        'reward/format_reward': [mean_format],
                        'reward/plan_reward': [mean_plan],
                        'reward/bonus_reward': [mean_bonus],
                        'terminate': sum(terminate)})
    print(f"Plan scores are {eval_df}")

    

    return eval_df





    