import sys
sys.path.append("/srv/chawak/planning-with-llms/src")
from shared import llm_utils
from prompting import completions_tracker

from unified_planning.shortcuts import *



class PolicyModel():

    def __init__(self):
        self.base_model,self.tokenizer=llm_utils.get_model_tokenizer()
        pass
    
    def Vanilla_one_shot(self,prompt,temp):
        i=0
        max_iter=2
        actions=[]
        #print(f'*************** PROMPT ***************: \n\n {prompt}')
        #keep looping until we get proper action strings
        while i<=max_iter and not actions:
            print(f'\n\n.......Querying LLM for a plan......... iteration: #{i}')
            # r = llm_utils.query_llm(prompt,model,temp)

            #debug querying with local model
            tokenized_input,processor= llm_utils.get_tokenized_input(prompt,self.base_model)
            r = llm_utils.Prompting_query_local_model(tokenized_input=tokenized_input,processor=processor,model=self.base_model)
            i+=1
            #print(f"\n\nLLM RESPONSE: {r}")
            actions=llm_utils.parse_action_tuples(r)
            
            if(actions):
                print(f"Model response in vanilla one shot: {r}")
                completions_tracker.track_completion(r)
        return actions,i
        
    def SFT_one_shot(self,prompt,model,temp=0):
        i=0
        max_iter=1
        actions=[]
        
        if model is None:
            print('No model has been passed!!!!!! using base model instead')
            model=self.base_model

        tokenized_input,processor=llm_utils.get_tokenized_input(prompt=prompt,model=model)
        while i<max_iter and not actions: 
            print(f'\n\n.......Querying LLM for a plan......... iteration: #{i}')
            r = llm_utils.query_local_model(tokenized_input=tokenized_input,processor=processor,model=model,temperature=temp)
            print(f'######################### Response from LLM:{r}')
            i+=1
            actions=llm_utils.parse_action_tuples(r)
        return actions,i                                
    
    def GRPO_one_shot(self,prompt,model,temp=0):
        i=0
        max_iter=3
        actions=[]
        
        if model is None:
            print('No model has been passed!!!!!! using base model instead')
            model=self.base_model

        tokenized_input,processor=llm_utils.get_tokenized_input(prompt=prompt,model=model)
        while i<max_iter and not actions: 
            print(f'\n\n.......Querying LLM for a plan......... iteration: #{i}')
            r = llm_utils.query_local_model(tokenized_input=tokenized_input,processor=processor,model=model,temperature=temp)
            print(f'######################### Response from LLM:{r}')
            i+=1
            actions=llm_utils.parse_action_tuples(r)

        return actions,i