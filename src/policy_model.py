from unified_planning.shortcuts import *
import utils
import prompts
import unifiedplanning_blocksworld as ubs

print('PRINTING GET BLOCKS FROM INSIDE POLICY MODEL',utils.blocks)

generation_prompt=prompts.get_generation_prompt(prompts.ic_gen_example,prompts.gs_gen_example,prompts.ic_gen_problem,prompts.gs_gen_problem)
response_gen = utils.query_llm(generation_prompt)

print('\n Planbench Generation-Prompt LLM response \n ',response_gen)

extracted_plan=utils.extract_plan(response_gen)
print('\n Extracted plan from Generation-Prompt response \n',extracted_plan)

# nextaction_prompt=prompts.get_nextaction_prompt(prompts.ic_nextaction_example,prompts.gs_nextaction_example,prompts.ic_nextaction_problem,prompts.gs_nextaction_problem)

# response_nextact = utils.query_llm(nextaction_prompt)
# print('\n Planbench Next-Action-Prompt LLM Response \n',response_nextact)

# next_action=utils.extract_next_action(response_nextact)
# print('\n Extracted next action : \n',next_action)