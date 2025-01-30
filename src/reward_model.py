import pickle, sys, copy, re, math
import pandas as pd
import re
import time
import numpy as np
import re
import policy_model as policy
import prompts
from unified_planning.shortcuts import *
from unified_planning.plans import *
import unifiedplanning_blocksworld as ubs

#DEFINE ubs problem
def get_blocks(predicate):
  blocks=[]
  for idx,word in enumerate(predicate):
      predicate[idx]=word.lower().strip(".,!? ()")
      if word.lower().strip(".,!? ")=="block":
          blocks.append(predicate[idx-1])
  return blocks

def parse_initial_condition(ic):
  #the blue block is clear, the yellow block is clear, the hand is empty, the blue block is on top of the orange block, the orange block is on top of the red block, the red block is on the table, the yellow block is on the table.

  predicate=ic.split(' ')
  blocks=get_blocks(predicate)
  #print('IC from Gen Prompt small loop %s' % (predicate))
  #print('\n blocks from Gen Prompt small loop',blocks,'\n')

  #make calls to unified planning and define initial conditions
  if 'clear' in predicate:
    #print('\nCLEAR was called\n')
    ubs.set_clear(blocks[0])

  if 'top' in predicate:
    #print('\TOP was called\n')
    ubs.set_on(blocks[0],blocks[1])

  if 'hand' in predicate:
    #print('\HAND was called\n')
    ubs.set_hand(True)
  
def parse_goal_state(gs):
#My goal is to have that: the red block is on top of the orange block, the blue block is on top of the yellow block.
  
  predicate=gs.split(' ')
  blocks=get_blocks(predicate)
  #print('GS from Gen Prompt small loop %s' % (predicate))
  #print('\n blocks from Gen Prompt small loop',blocks,'\n')

  #make calls to unified planning and define goal state
  if 'top' in predicate:
    #print('\TOP was called\n')
    ubs.set_on_goal(blocks[0],blocks[1])

problem_string_ic=prompts.ic_gen_problem
problem_string_gs=prompts.gs_gen_problem

[parse_initial_condition(ics) for ics in problem_string_ic.split(',')]
[parse_goal_state(gss) for gss in problem_string_gs.split(',')]

print('\n UBS Problem state: \n',ubs.problem)

#DEFINE ubs plan
def parse_plan(action):
  predicate=action.split(' ')
  print('ÄFTER SPLITTING',predicate)

  for word in predicate:
      blocks=[]
      if word.lower().strip('()')=='unstack':
        print('\n UNSTACK was called \n')
        blocks=[predicate[1],predicate[2].strip(')')]
        ubs.call_func('unstack',blocks)

      if word.lower().strip('()')=='put-down':
        print('\n PUT-DOWN was called\n')
        blocks=[predicate[1].strip(')')]
        ubs.call_func('put-down',blocks)

      if word.lower().strip('()')=='pick-up':
         print('\n PICK-UP was called\n')
         blocks=[predicate[1].strip(')')]
         ubs.call_func('pick-up',blocks)

      if word.lower().strip('()')=='stack':
         print('\n STACK was called\n')
         blocks=[predicate[1],predicate[2].strip(')')]
         ubs.call_func('stack',blocks)
      print(blocks)

#['(unstack blue orange)', '(put-down blue)', '(pick-up red)', '(stack red orange)', '(pick-up blue)', '(stack blue yellow)']
[parse_plan(action) for action in policy.extracted_plan]

#(pick-up red)
#parse_plan(policy.next_action)

print('\n UBS model plan: \n',ubs.model_plan)

gen_result=ubs.solver_sol()

def validation(plan):
  if ubs.validator.validate(ubs.problem, plan):
    return('The plan is valid')
  else:
    return('The plan is invalid')
  
print('\nSOLVER SOLUTION:',gen_result)

print('Validation on Model plan:',validation(ubs.model_plan))

print('Validation on Pyperplan Solver:',validation(gen_result))