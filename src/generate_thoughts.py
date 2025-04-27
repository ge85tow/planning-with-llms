import unifiedplanning_blocksworld as bw
import prompts
import planbench as pb
import policy_model as policy
import llm_utils
import data_loader
import pandas as pd
import regex as re
from unified_planning.shortcuts import *
from unified_planning.plans import *
from importlib import reload
import warnings 
warnings.filterwarnings('ignore')

def get_deep_thoughts(df,model,temp):
    print('Entering main function')
    thoughts=[]
    for i in range(len(df)):
        print(f'\n\n--------------------Entering main loop PROBLEM #{i}--------------------')
        #prompt,init,goal=prompts.make_prompt(df.iloc[i]['init'],df.iloc[i]['goal'],df.iloc[i]['demo_init'],df.iloc[i]['demo_goal'],df.iloc[i]['demo_plan'])
        prompt=df['prompt'][i]
        demo_prompt=prompt.split('[PLAN]')[0].strip()
        demo_prompt=demo_prompt+'[PLAN]'
        r=llm_utils.query_llm(prompt=demo_prompt+"Let's DeepThink this.",model=model,temperature=temp,max_tokens=1200)
        thought=r.split('<think>')[1].split('</think>')[0].strip()
        print(f'Thought #{i}: {thought}')
        thoughts.append(thought)
    return thoughts
n=3
split='train'
model='deepseek-ai/DeepSeek-R1-Distill-Llama-70B'
temp=0
#df=pd.read_csv(f'../data/{n}_blocks/SFT_{split}_{n}_blocks_fullPlan').drop(columns='Unnamed: 0')
df=pd.read_csv(f'../data/{n}_blocks/SFT_contaminated_test_{n}_blocks_fullPlan').drop(columns='Unnamed: 0')
thoughts=get_deep_thoughts(df,model,temp)
thoughts_df=pd.DataFrame(thoughts,columns=['thought'])
thoughts_df.to_csv(f'../data/{n}_blocks/contaminated_test_thoughts.csv')
