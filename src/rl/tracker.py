format_scores=[]
plan_scores=[]
bonus_scores=[]
total_completions_seen=0
# best_scores=[]

def accumulate_format_reward(scores):
    print(f"Type: {type(scores)}, Device: {getattr(scores, 'device', 'N/A')}")
    format_scores.extend(scores)

def accumulate_plan_reward(scores):
    print(f"Type: {type(scores)}, Device: {getattr(scores, 'device', 'N/A')}")
    plan_scores.extend(scores)

def accumulate_bonus_reward(scores):
    print(f"Type: {type(scores)}, Device: {getattr(scores, 'device', 'N/A')}")
    bonus_scores.extend(scores)