import pickle, sys, copy, re, math
import pandas as pd
import concurrent.futures
import re
import time
import numpy as np
import re
from anytree import Node,RenderTree
import prompts

blocks=set()

def get_blocks(statement):
    problem=statement.split(' ')
    blocks=set()
    for idx,word in enumerate(problem):
        if word =='block':
            blocks.add(problem[idx-1])
    #print(blocks)
    return blocks

blocks=get_blocks(prompts.ic_gen_problem)

#------------------------------------PARSING FUNCTIONS-------------------------------

#returns action strings' list from plan
def extract_plan_action_strings(plan_output):
    actions = plan_output.split('[PLAN]')[1].split('[PLAN END]')[0].strip().split('\n')
    actions = [a.strip() for a in actions]
    return actions

#returns next action 
def extract_next_action_string(plan_output):
    return plan_output.split('[NEXT ACTION]')[1].split('[END NEXT ACTION]')[0].strip()

#formats action strings 
def parse_action(action):
    predicate = action.split(' ')[0]
    if predicate.lower() == 'unstack':
        pattern = r"unstack the (\w+) block from on top of the (\w+) block"
        replacement = r"(unstack \1 \2)"
        return re.sub(pattern, replacement, action)
    if predicate.lower() == 'stack':
        pattern = r"stack the (\w+) block on top of the (\w+) block"
        replacement = r"(stack \1 \2)"
        return re.sub(pattern, replacement, action)
    if predicate.lower() == 'put':
        pattern = r"put down the (\w+) block"
        replacement = r"(put-down \1)"
        return re.sub(pattern, replacement, action)
    if predicate.lower() == 'pick':
        pattern = r"pick up the (\w+) block"
        replacement = r"(pick-up \1)"
        return re.sub(pattern, replacement, action)

def extract_plan(plan_output):
    actions = extract_plan_action_strings(plan_output)
    parsed_actions = [parse_action(a) for a in actions]
    return parsed_actions

def extract_next_action(plan_output):
    return parse_action(extract_next_action_string(plan_output))

#--------------------------query llm-------------------------------
from openai import OpenAI
openai=OpenAI(
    api_key="qQSK4UL7SbIzA1ipuzCCNoItChpioZAv",
    base_url="https://api.deepinfra.com/v1/openai",
)

chat_completion = openai.chat.completions.create(
    model="meta-llama/Meta-Llama-3-70B-Instruct",
    messages=[
        {"role": "user", "content": "Hi!"},],
)
print(chat_completion.choices[0].message.content)

def query_llm(prompt,temperature=0.7, max_tokens=100):
  chat_completion = openai.chat.completions.create(
    model="meta-llama/Meta-Llama-3-70B-Instruct",
    messages=[
        {"role": "user", "content": prompt},],
    temperature=temperature,
    )
  return chat_completion.choices[0].message.content