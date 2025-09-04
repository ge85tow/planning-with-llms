import regex as re
from rl import tracker
from verl_env_wrapper import BlocksworldProblem  # See note below

# --- Format & Plan Reward Functions ---

gibberish = r"(.*)"
plan = r"\[PLAN\](.*?)\[PLAN END\]"
think = r"<think>(.*?)<\/think>"

def get_score(response):
    response = response.strip()
    score = 0

    #case-1: no gibberish, no plan, yes think 
    pattern1 = f"{think}"
    compiled1 = re.compile(pattern1, re.DOTALL)

    #case-2: no gibberish, yes plan, no think
    pattern2 = f"{plan}"
    compiled2 = re.compile(pattern2, re.DOTALL)

    #case-3: no gibberish, yes plan, yes think
    pattern3 = f"{think}\s*{plan}"
    compiled3 = re.compile(pattern3, re.DOTALL)

    #case-6: yes gibberish, yes plan, yes think
    pattern6 = f"{think}{gibberish}{plan}"
    compiled6 = re.compile(pattern6, re.DOTALL)

    fullmatchthink = False
    if compiled1.fullmatch(response):
        score = 7
        fullmatchthink = True
    if compiled1.search(response) and not fullmatchthink:
        score = 5

    fullmatchplan = False
    if compiled2.fullmatch(response):
        score = 3
        fullmatchplan = True
    if compiled2.search(response) and not fullmatchplan:
        score = 2

    fullmatchcombo = False
    if compiled3.fullmatch(response):
        score = 20
        fullmatchcombo = True
    if compiled6.search(response) and not fullmatchcombo:
        score = 15

    return score

def format_reward(completions, **kwargs):
    import os
    print(f'memory debug in reward/format_reward after {tracker.total_completions_seen} problems: {os.system("nvidia-smi")}')
    scores = []
    responses = [completion for completion in completions]
    for response in responses:
        response = response.strip()
        print(f'{tracker.total_completions_seen}. In format reward, model response is: {repr(response)}')
        score = get_score(response)
        scores.append(score)
        print(f"Format reward for this response is: {score}")
    tracker.accumulate_format_reward(scores)
    return scores

# --- Plan reward logic ---

def make_action_tuples(response):
    from shared import llm_utils
    action_tuples = llm_utils.parse_action_tuples(response)
    if not action_tuples:
        print(f'\n\n Imparsable response')
        return False
    return action_tuples

def response2plan(problem, init, goal, response):
    from shared import prompts, planbench as pb
    model_plan = None
    init = prompts.parse_init(init)
    goal = prompts.parse_goal(goal)
    pb.parse_planbench_initial_condition(problem, init)
    pb.parse_planbench_goal_state(problem, goal)
    print(f'\n\nBlocksworld Problem Initial Values:{problem.initial_values}')
    print(f'\nBlocksworld Problem Goal State:{problem.goals}')
    action_tuples = make_action_tuples(response)
    if action_tuples:
        model_plan = problem.GRPOcreate_plan_from_tuples(action_tuples)
    print(f"Model responded with this plan: {model_plan}")
    return action_tuples, model_plan

def apply_plan(problem, model_plan):
    simulation = problem.create_seq_simulation()
    va_counter = 0
    valid_state, va_counter, distance2goal, flag = problem.GRPO_check_and_apply(simulation, model_plan)
    d = problem.actions_to_goal(valid_state)
    distance2goal.append(d)
    return valid_state, va_counter, distance2goal

def goal_proximity(distance2goal: list) -> list:
    scores = []
    for idx, d in enumerate(distance2goal):
        if d == 0:
            break
        if idx <= len(distance2goal)-2:
            d_old = d
            d_new = distance2goal[idx+1]
            score = max(0, (d_old-d_new)*5)
            scores.append(score)
    return scores

def get_plan_len(plan):    
    return len(plan.split('\n'))-2

def gold_plan_reward(problem, gold_plan):
    print('~'*70,'\n ATTENTION: THIS IS ALL FOR GOLD PLAN REWARD')
    score = 0
    action_tuples = make_action_tuples(gold_plan)
    gold_plan_ob = problem.GRPOcreate_plan_from_tuples(action_tuples)
    current_state, va_counter, distance2goal = apply_plan(problem, gold_plan_ob)
    score += va_counter*2
    distance_scores = goal_proximity(distance2goal)
    score += sum(distance_scores)
    print('GOLD PLAN SCORE IS: ',score)
    print('~'*70, '\n END ATTENTION: GOLD PLAN REWARD COMPUTE END')
    return score

def bonus_reward(problem, valid_state, plan_len, goldplanlen):
    score = 0
    if problem.terminate(valid_state):
        score += 20
        if plan_len == goldplanlen: score += 10
        if plan_len < goldplanlen: score += 15
    return score

def response_score(response, init, goal, gold_plan):
    print('-'*70,f"Entering response score for completion {tracker.total_completions_seen}",'-'*70)
    print(f'\n Model Response: {response}')
    problem = BlocksworldProblem()
    score = 0
    bonus_score = 0
    action_tuples, model_plan = response2plan(problem=problem, init=init, goal=goal, response=response)
    if model_plan:
        current_state, valid_action_count, distance2goal = apply_plan(problem=problem, model_plan=model_plan)
        print(f'Distance 2 goal metric:{distance2goal}')
        gold_plan_len = get_plan_len(gold_plan)
        score += min(gold_plan_len*2, valid_action_count*2)
        print(f"Valid actions score is: {score}")
        distance_scores = goal_proximity(distance2goal)
        print(f"Proximity scores are: {distance_scores}")
        score += sum(distance_scores)
        print(f"Score without bonus reward: {score}")
        gold_plan_score = gold_plan_reward(problem=problem, gold_plan=gold_plan)
        score = round((score/gold_plan_score)*50, 2)
        bonus_score = bonus_reward(problem, current_state, len(action_tuples), gold_plan_len)
        print(f"Score with bonus reward: {score+bonus_score}")
    tracker.total_completions_seen += 1
    print('!'*80,f"FINAL SCORE FOR THIS GENERATION: {score+bonus_score} \n\n\n")
    return score, bonus_score

def responses_scores(response_list, init_list, goal_list, gold_plan_list):
    plan_scores = []
    bonus_scores = []
    for idx in range(len(response_list)):
        plan_score, bonus_score = response_score(
            response=response_list[idx],
            init=init_list[idx],
            goal=goal_list[idx],
            gold_plan=gold_plan_list[idx])
        plan_scores.append(plan_score)
        bonus_scores.append(bonus_score)
    return plan_scores, bonus_scores

def plan_reward(completions, init, goal, gold_plan, **kwargs):
    responses = [completion for completion in completions]
    init_list = init
    goal_list = goal
    gold_plan_list = gold_plan
    scores = responses_scores(response_list=responses, init_list=init_list, goal_list=goal_list, gold_plan_list=gold_plan_list)
    plan_scores = scores[0]
    bonus_scores = scores[1]
    tracker.accumulate_plan_reward(plan_scores)
    tracker.accumulate_bonus_reward(bonus_scores)
    added_scores = [plan + bonus for plan,bonus in zip(plan_scores, bonus_scores)]
    return added_scores