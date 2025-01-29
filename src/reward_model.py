import pickle, sys, copy, re, math
import pandas as pd
import re
import time
import numpy as np
import re
import policy_model as policy
from unified_planning.shortcuts import *
from unified_planning.plans import *
import unifiedplanning_blocksworld as ubs
import prompts


def parse_initial_condition(ic):
  #the blue block is clear, the yellow block is clear, the hand is empty, the blue block is on top of the orange block, the orange block is on top of the red block, the red block is on the table, the yellow block is on the table.
  predicate=ic.split(' ')
  #print('IC from Gen Prompt small loop %s' % (predicate))
  blocks=[]
  for idx,word in enumerate(predicate):
      predicate[idx]=word.lower().strip(".,!? ")


      if word.lower().strip(".,!? ")=="block":
          blocks.append(predicate[idx-1])
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
  blocks=[]
  #print('GS from Gen Prompt small loop %s' % (predicate))
  for idx,word in enumerate(predicate):
      predicate[idx]=word.lower().strip(".,!? ")

      if word.lower().strip(".,!? ")=="block":
          blocks.append(predicate[idx-1])
  #print('\n blocks from Gen Prompt small loop',blocks,'\n')

  #make calls to unified planning and define goal state
  if 'top' in predicate:
    print('\TOP was called\n')
    ubs.set_on_goal(blocks[0],blocks[1])



#add parsed conditions to ubs problem
for ics in prompts.ic_gen_problem.split(','): 
  #count=0
  #print('\n%d####. ICS from Gen Prompt BIG LOOP: %s\n' % (count,ics)) 
  parse_initial_condition(ics)
  #count+=1

for gss in prompts.gs_gen_problem.split(','):
  #count=0
  #print('\n%d####. GSS from Gen Prompt BIG LOOP: %s\n' % (count,gss))
  parse_goal_state(gss)
  #count+=1

print('\n UBS Problem state: \n',ubs.problem)

model_solution=SequentialPlan(ubs.problem.)
#model_solution.add_objects(ubs.blocks.values())
#model_solution.add_fluents(ubs.Problem.fluents)
#model_solution.add_actions(ubs.Problem.actions)
#model_solution=policy.extracted_plan


def validation(plan):
  if ubs.validator.validate(ubs.problem, plan):
    print('The plan is valid')
  else:
    print('The plan is invalid')

print(validation(model_solution))