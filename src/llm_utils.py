import pickle, sys, copy, re, math
import concurrent.futures
import regex as re
import time
import numpy as np
import re
from anytree import Node,RenderTree
import prompts
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from huggingface_hub import login


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
            m = re.search(r"[Uu]nstack (?:the )?(\w+) block (?:from on top of|from) (?:the )?(\w+) block", action)
            if m:
                result = ('unstack', m.group(1), m.group(2))
                #print(f'\nFor predicate: {predicate} the result is : {result}')

        elif predicate.lower() == 'stack':
            m = re.search(r"[Ss]tack (?:the )?(\w+) block (?:on top of|top of|on) (?:the )?(\w+) block", action)
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

#deep-infra API initialization
from openai import OpenAI
openai=OpenAI(
    #api_key="qQSK4UL7SbIzA1ipuzCCNoItChpioZAv",
    api_key="XqPUITUxARnzuvDPvLWMQW86dKIAS1Ih",
    base_url="https://api.deepinfra.com/v1/openai",
)

def query_llm(prompt,model="meta-llama/Meta-Llama-3-70B-Instruct",temperature=0.7, max_tokens=None,):
  print(f'Querying model : {model} with temperature: {temperature}')
  chat_completion = openai.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": prompt}],
    temperature=temperature,
    max_tokens=max_tokens
    )
  return chat_completion.choices[0].message.content

#local model initialization
login(token="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")
cache_dir='/home/chawak/models/huggingface'

def get_model_tokenizer(name='google/gemma-2-9b-it'):
    tokenizer = AutoTokenizer.from_pretrained(name, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        name,
        cache_dir=cache_dir,
        device_map="auto",
        #device_map='auto',
        torch_dtype=torch.bfloat16,
    )
    return model, tokenizer

def make_chat_format(prompt):
    chat_prompt=[{"role": "user", "content": prompt}]
    return chat_prompt

def format_prompt_for_tokenizer(chat_prompt):
    # Convert chat format to a single string
    prompt_str = ""
    for message in chat_prompt:
        role = message["role"]
        content = message["content"]
        prompt_str += f"{role}: {content}\n"
    return prompt_str

def query_local_model(prompt,tokenizer,model='google/gemma-2-9b-it',model_name='google/gemma-2-9b-it',temperature=0):
    
    #format our prompt to make it chat-style
    chat_prompt=make_chat_format(prompt)
    prompt=format_prompt_for_tokenizer(chat_prompt)
    print(f'PROMPT: {prompt}')
    #tokenize prompt and fetch encoded input and its length
    inputs=tokenizer(prompt, return_tensors='pt').to('cuda:3')
    input_len=inputs['input_ids'].shape[1]
    #get llm to generate response on tokenized prompt
    print(f'----------Querying model {model_name}: with temperature: {temperature}----------')
    response=model.generate(**inputs,
                            temperature=temperature,
                            max_new_tokens=256,
                            #early_stopping=False,
                            #do_sample=False,
                            #eos_token_id=tokenizer.eos_token_id,
                            #pad_token_id=tokenizer.pad_token_id,
                            )
    #decode response from LLM
    decodeop=tokenizer.decode(response[0][input_len:],skip_special_tokens=True)
    print(f'Response from LLM : {decodeop}')
    return decodeop

'''[PLAN]  
1. Unstack the yellow block from on top of the red block.
2. Stack the yellow block on top of the blue block.
3. Unstack the red block from on top of the orange block.
4. Stack the red block on top of the yellow block.
5. Stack the yellow block on top of the blue block.

[PLAN END]
'''

