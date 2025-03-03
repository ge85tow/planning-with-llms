import pickle, sys, copy, re, math
import concurrent.futures
import regex as re
import time
import numpy as np
import re
from anytree import Node,RenderTree
import prompts

#------------------------------------ LLM OUTPUT PARSING FUNCTIONS-------------------------------

# returns action strings' list from generated plan
def extract_plan_action_strings(plan_output):
    if '[PLAN]' not in plan_output or '[PLAN END]' not in plan_output:
        return False
    actions = plan_output.split('[PLAN]')[1].split('[PLAN END]')[0].strip().split('\n')
    actions = [a.strip() for a in actions]
    return actions

# returns next action from generated plan
def extract_next_action_string(plan_output):
    return plan_output.split('[NEXT ACTION]')[1].split('[END NEXT ACTION]')[0].strip()

#def form_action_tuple(action):
    #print(f"After EXTRCAT PLAN ACTIONS: {actions}")
    predicate = action.split(' ')[0]
    if predicate.lower() == 'unstack':
        m = re.match(r"unstack the (\w+) block from on top of the (\w+) block", action)
        if m:
            return ('unstack', m.group(1), m.group(2))
        return False
    elif predicate.lower() == 'stack':
        m = re.match(r"stack the (\w+) block on top of the (\w+) block", action)
        if m:
            return ('stack', m.group(1), m.group(2))
        return False
    elif predicate.lower() == 'put':
        m = re.match(r"put down the (\w+) block", action)
        if m:
            return ('put-down', m.group(1))
        return False
    elif predicate.lower() == 'pick':
        m = re.match(r"pick up the (\w+) block", action)
        if m:
            return ('pick-up', m.group(1))
        return False
    else:
       print(f'cannot detect predicate here: {action}')

#parses an action and returns action tuple
def form_action_tuple(action):

    seperators=r"[ .]"
    predicate = re.split(seperators,action)
    list_predicate = list(filter(None, predicate))
    #print(f'\n\n Action:{action} after seperators:{list_predicate}')
    
    flag=False
    result=None

    for i,predicate in enumerate(list_predicate):

      #print(f'\n\n Action: {action}, predicate: {predicate.lower()}, counter: {i}')

      if(predicate.lower() in {'unstack','stack','put','pick'}):
        #print('ENTERING CASES')
        flag=True

        if predicate.lower() == 'unstack':
            m = re.search(r"[Uu]nstack (?:the )?(\w+) block from on top of (?:the )?(\w+) block", action)
            if m:
                result = ('unstack', m.group(1), m.group(2))
                #print(f'\nFor predicate: {predicate} the result is : {result}')

        elif predicate.lower() == 'stack':
            m = re.search(r"[Ss]tack (?:the )?(\w+) block on top of (?:the )?(\w+) block", action)
            if m:
                result = ('stack', m.group(1), m.group(2))
                #print(f'\nFor predicate: {predicate} the result is : {result}')

        elif predicate.lower() == 'put':
            m=re.search(r"[pP]ut[- ]down (?:the )?(\w+) block",action)
            if m:
                result = ('put-down', m.group(1))
                #print(f'\nFor predicate: {predicate} the result is : {result}')

        elif predicate.lower() == 'pick':
            m = re.search(r"[pP]ick[- ]up (?:the )?(\w+) block", action)
            if m:
                result = ('pick-up', m.group(1))
                #print(f'\nFor predicate: {predicate} the result is : {result}')
        
        if result:
          break

    if(not flag):
      print(f'cannot detect predicate here: {action}')

    return result if result else False

# parse set of tuples from generated plan
def parse_action_tuples(plan_output):
    actions = extract_plan_action_strings(plan_output)
    print(f"\nAfter EXTRCAT PLAN ACTIONS: {actions}")
    if not actions:
        return actions
    else:
        action_tuples = [form_action_tuple(a) for a in actions]
        return action_tuples

# parse next action tuple from generated plan
def parse_next_action_tuple(plan_output):
    return form_action_tuple(extract_next_action_string(plan_output))



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

'''[PLAN]  
1. Unstack the yellow block from on top of the red block.
2. Stack the yellow block on top of the blue block.
3. Unstack the red block from on top of the orange block.
4. Stack the red block on top of the yellow block.
5. Stack the yellow block on top of the blue block.

[PLAN END]
'''