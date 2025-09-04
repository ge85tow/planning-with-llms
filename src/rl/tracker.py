import pandas as pd
format_scores=[]
plan_scores=[]
terminate=[]
total_completions_seen=0


def accumulate_format_reward(scores):
    print(f"Type: {type(scores)}, Device: {getattr(scores, 'device', 'N/A')}")
    format_scores.extend(scores)

def accumulate_plan_reward(scores):
    print(f"Type: {type(scores)}, Device: {getattr(scores, 'device', 'N/A')}")
    plan_scores.extend(scores)

def accumulate_terminate(scores):
    print(f"Type: {type(scores)}, Device: {getattr(scores, 'device', 'N/A')}")
    terminate.extend(scores)
    