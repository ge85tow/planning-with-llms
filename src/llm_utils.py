import pickle, sys, copy, re, math
import concurrent.futures
import re
import time
import numpy as np
import re
from anytree import Node,RenderTree
import prompts

#------------------------------------ LLM OUTPUT PARSING FUNCTIONS-------------------------------

# returns action strings' list from generated plan
def extract_plan_action_strings(plan_output):
    actions = plan_output.split('[PLAN]')[1].split('[PLAN END]')[0].strip().split('\n')
    actions = [a.strip() for a in actions]
    return actions

# returns next action from generated plan
def extract_next_action_string(plan_output):
    return plan_output.split('[NEXT ACTION]')[1].split('[END NEXT ACTION]')[0].strip()

# creates individual predicate argument tuple 
def form_action_tuple(action):
    predicate = action.split(' ')[0]
    if predicate.lower() == 'unstack':
        m = re.match(r"unstack the (\w+) block from on top of the (\w+) block", action)
        return ('unstack', m.group(1), m.group(2))
    elif predicate.lower() == 'stack':
        m = re.match(r"stack the (\w+) block on top of the (\w+) block", action)
        return ('stack', m.group(1), m.group(2))
    elif predicate.lower() == 'put':
        m = re.match(r"put down the (\w+) block", action)
        return ('put-down', m.group(1))
    elif predicate.lower() == 'pick':
        pattern = re.match(r"pick up the (\w+) block", action)
        return ('pick-up', m.group(1))
    else:
       print(f'cannot detect predicate here: {action}')

# parse set of tuples from generated plan
def parse_action_tuples(plan_output):
    actions = extract_plan_action_strings(plan_output)
    action_tuples = [form_action_tuple(a) for a in actions]
    return action_tuples

# parse next action tuple from generated plan
def parse_next_action_tuple(plan_output):
    return parse_action_tuple(extract_next_action_string(plan_output))



#--------------------------query llm-------------------------------

from openai import OpenAI
openai=OpenAI(
    api_key="qQSK4UL7SbIzA1ipuzCCNoItChpioZAv",
    base_url="https://api.deepinfra.com/v1/openai",
)

def query_llm(prompt,temperature=0.7, max_tokens=100):
  chat_completion = openai.chat.completions.create(
    model="meta-llama/Meta-Llama-3-70B-Instruct",
    messages=[
        {"role": "user", "content": prompt},],
    temperature=temperature,
    )
  return chat_completion.choices[0].message.content