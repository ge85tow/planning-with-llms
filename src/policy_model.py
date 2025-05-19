from unified_planning.shortcuts import *
import llm_utils
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

r='''[PLAN]  
            1. Unstack the yellow block from on top of the red block.
            2. Stack the yellow block on top of the blue block.
            3. Unstack the red block from on top of the orange block.
            4. Stack the red block on top of the yellow block.
            5. Stack the yellow block on top of the blue block.
            6. Pick up the yellow block.

            [PLAN END]'''

class PolicyModel():

    def __init__(self):
        #self.base_model,self.tokenizer=llm_utils.get_model_tokenizer()
        pass
    # def next_action_one_shot(self, 
    #                          problem_init, problem_goal,problem_action_hist,  # the state S
    #                          example_init, example_goal,example_action_hist,example_next_action): # Few-shot guidance
        
    #     nextaction_prompt=prompts.get_nextaction_prompt(example_init, example_goal,example_action_hist,example_next_action,
    #                                                     problem_init, problem_goal,problem_action_hist)
    #     #print("\nPROMPT:\n",nextaction_prompt)
    #     r = llm_utils.query_llm(nextaction_prompt)
    #     print("\nRESPONSE:\n",r)
    #     action_tuple = llm_utils.parse_next_action_tuple(r)
    #     return action_tuple
    
    def Vanilla_fullSol_one_shot(self,prompt,model,temp):
        i=0
        max_iter=2
        actions=[]
        print(f'*************** PROMPT ***************: \n\n {prompt}')
        #keep looping until we get proper action strings
        while i<=max_iter and not actions:
            print(f'\n\n.......Querying LLM for a plan......... iteration: #{i}')
            r = llm_utils.query_llm(prompt,model,temp)
            i+=1
            #print(f"\n\nLLM RESPONSE: {r}")
            actions=llm_utils.parse_action_tuples(r)
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