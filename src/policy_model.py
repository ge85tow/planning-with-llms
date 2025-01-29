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

'''ic_nextaction_example="the red block is clear, the orange block is clear, the yellow block is clear, the hand is empty, the orange block is on top of the blue block, the red block is on the table, the blue block is on the table and the yellow block is on the table."
gs_nextaction_example="the red block is on top of the blue block and the yellow block is on top of the red block and the orange block is on the table."

ic_nextaction_problem="the blue block is clear, the yellow block is clear, the hand is empty, the blue block is on top of the orange block, the orange block is on top of the red block, the red block is on the table, the yellow block is on the table"
gs_nextaction_problem="the red block is on top of the orange block and the blue block is on top of the yellow block"

nextaction_prompt=prompts.get_nextaction_prompt(ic_nextaction_example,gs_nextaction_example,ic_nextaction_problem,gs_nextaction_problem)

response_nextact = utils.query_llm(nextaction_prompt)
print('\n Planbench Next-Action-Prompt LLM Response \n',response_nextact)
print('\n Extracted next action : \n',utils.extract_next_action(response_nextact))'''