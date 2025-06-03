import pickle, sys, copy, re, math
import pandas as pd
import re
import time
import numpy as np
import re
import shared.policy_model as policy
import shared.prompts as prompts
from unified_planning.shortcuts import *
from unified_planning.plans import *
import shared.unifiedplanning_blocksworld as ubs

#DEFINE ubs problem

#DEFINE ubs plan


#['(unstack blue orange)', '(put-down blue)', '(pick-up red)', '(stack red orange)', '(pick-up blue)', '(stack blue yellow)']
#[parse_plan(action) for action in policy.extracted_plan]

#(pick-up red)
#parse_plan(policy.next_action)

#print('\n UBS model plan: \n',ubs.model_plan)

#gen_result=ubs.solver_sol()

#def validation(plan):
#  if ubs.validator.validate(ubs.problem, plan):
#    return('The plan is valid')
#  else:
#    return('The plan is invalid')
  
#print('\nSOLVER SOLUTION:',gen_result)

#print('Validation on Model plan:',validation(ubs.model_plan))

#print('Validation on Pyperplan Solver:',validation(gen_result))