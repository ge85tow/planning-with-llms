from unified_planning.shortcuts import *
import llm_utils
import prompts
import unifiedplanning_blocksworld as ubs


#print('PRINTING GET BLOCKS FROM INSIDE POLICY MODEL',utils.blocks)

#generation_prompt=prompts.get_generation_prompt(prompts.ic_gen_example,prompts.gs_gen_example,prompts.ic_gen_problem,prompts.gs_gen_problem)
#response_gen = utils.query_llm(generation_prompt)

#print('\n Planbench Generation-Prompt LLM response \n ',response_gen)

#extracted_plan=utils.extract_plan(response_gen)
#print('\n Extracted plan from Generation-Prompt response \n',extracted_plan)


# response_nextact = utils.query_llm(nextaction_prompt)
# print('\n Planbench Next-Action-Prompt LLM Response \n',response_nextact)

# next_action=utils.extract_next_action(response_nextact)
# print('\n Extracted next action : \n',next_action)

class PolicyModel():

    def __init__(self):
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
    
    def Vanilla_fullSol_one_shot(self,prompt):
        i=0
        max_iter=2
        actions=[]
        while i<=max_iter and not actions:
            print(f'\n\nIteration: #{i}...............QUERYING LLM for a plan')
            r = llm_utils.query_llm(prompt)
            i+=1
            print(f"\n\nLLM RESPONSE: {r}")
            actions=llm_utils.parse_action_tuples(r)

            return actions