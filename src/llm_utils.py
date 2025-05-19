import pickle, sys, copy, re, math
import concurrent.futures
import regex as re
import time
import numpy as np
import re
from anytree import Node,RenderTree
import prompts
from transformers import AutoTokenizer,AutoProcessor,Gemma3ForCausalLM
import torch
from huggingface_hub import login
from datasets import load_from_disk
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
login(token="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")

#------------------------------------ LLM OUTPUT PARSING FUNCTIONS-------------------------------

# returns action strings' list from generated plan
def extract_plan_action_strings(plan_output):
    if '[PLAN]' not in plan_output or '[PLAN END]' not in plan_output:
        return False
    actions = plan_output.split('[PLAN]')[1].split('[PLAN END]')[0].strip().split('\n')
    actions = [a.strip() for a in actions]
    actions = [a for a in actions if a != 'model']
    return actions

# returns next action from generated plan
def extract_next_action_string(plan_output):
    return plan_output.split('[NEXT ACTION]')[1].split('[END NEXT ACTION]')[0].strip()

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
            m = re.search(r"[Uu]nstack (?:the )?(\w+) (?:block )?(?:from on top of|from) (?:the )?(\w+)(?: block)?", action)
            if m:
                result = ('unstack', m.group(1), m.group(2))
                #print(f'\nFor predicate: {predicate} the result is : {result}')

        elif predicate.lower() == 'stack':
            m = re.search(r"[Ss]tack (?:the )?(\w+) (?:block )?(?:on top of|top of|on) (?:the )?(\w+)(?: block)?", action)
            if m:
                result = ('stack', m.group(1), m.group(2))
                #print(f'\nFor predicate: {predicate} the result is : {result}')

        elif predicate.lower() == 'put':
            m=re.search(r"[pP]ut[- ]down (?:the )?(\w+)(?: block)?",action)
            if m:
                result = ('put-down', m.group(1))
                #print(f'\nFor predicate: {predicate} the result is : {result}')

        elif predicate.lower() == 'pick':
            m = re.search(r"[pP]ick[- ]up (?:the )?(\w+)(?: block)?", action)
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

def tokenize_input(tokenizer,model, input): 
    inputs=tokenizer(input, return_tensors='pt').to(model.device)
    return inputs

def load_tokenized_data(n):
    
    #load train dataset
    split='train'
    data_path=f'../data/{n}_blocks/tokenized_dataset/{split}'

    #change for toy example
    #data_path=f'/srv/chawak/planning-with-llms/data/toy_label_nopad/{split}'

    train_data=load_from_disk(data_path)

    #debug 
    #train_data=train_data.select(range(1))
    # print(f'The train data is: {train_data}')
    # print(f'------------------Train data------------------')
    # print(f"Input ids tokens: {train_data['input_ids'][0]}")
    # print(f"Length of input ids tokens: {len(train_data['input_ids'][0])}")
    # print(f"Label tokens: {train_data['labels'][0]}")
    # print(f"Length of label tokens: {len(train_data['labels'][0])}")   

    #load evaluation dataset
    split='val'
    data_path=f'../data/{n}_blocks/tokenized_dataset/{split}'

    #change for toy example
    #data_path=f'/srv/chawak/planning-with-llms/data/toy_label_nopad/{split}'

    eval_data=load_from_disk(data_path)
    #eval_data=eval_data.select(range(1))
    #print(f'The eval data is:{eval_data}')
    return train_data, eval_data
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

#-------- local model ----------
login(token="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")
cache_dir='/home/chawak/huggingface'

def get_model_tokenizer(name='google/gemma-3-12b-it'):

    tokenizer = AutoTokenizer.from_pretrained(name, cache_dir=cache_dir)
    model = Gemma3ForCausalLM.from_pretrained(
        name,
        cache_dir=cache_dir,
        device_map="auto",
        #gradient_checkpointing=True,
        torch_dtype=torch.bfloat16,
        attn_implementation='eager'
    )
    return model, tokenizer

processor = AutoProcessor.from_pretrained('google/gemma-3-12b-it')
def get_tokenized_input(prompt,model):
    
    #format our prompt to make it chat-style
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text":prompt}
            ]
        }
        ]

    #print chat prompt for sanity-check
    chat_prompt=processor.apply_chat_template(
        messages,
        tokenize=False,
    )
    print(f'Prompt in the chat-template:{chat_prompt}')

    #tokenize input to model
    tokenized_input = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt").to(model.device, dtype=torch.bfloat16)
    
    return tokenized_input,processor

def query_local_model(tokenized_input,processor,model='google/gemma-3-12b-it',temperature=0):
   
    #print(f'Tokenized input structure: {tokenized_input}')
    #get input length
    input_len=len(tokenized_input['input_ids'][0])
    #tags
    input_len-=11
    #print(f'input length: {input_len}')
    #get llm to generate response on tokenized prompt
    filtered_inputs={k:v for k,v in tokenized_input.items() if k!= 'token_type_ids'}
    #saving GPU memory 
    
    with torch.inference_mode():
        generation = model.generate(
                        **filtered_inputs,
                        max_new_tokens=1024,
                        temperature=temperature)
        generation = generation[0][input_len:]
        decoded = processor.decode(generation, skip_special_tokens=True)
    return decoded



