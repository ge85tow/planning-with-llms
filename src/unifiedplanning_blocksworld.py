# -*- coding: utf-8 -*-
from unified_planning.shortcuts import *
from unified_planning.plans import *
import utils

#In initial state, declrations for on the table blocks??


# Define the problem
problem = Problem("BlocksWorld")

Block = UserType("Block")

# Declare dictionary of block objects
blocks={}
print('PRINTING prompts.blocks from INSIDE UBS',utils.blocks)
for block_element in utils.blocks:  
  #print('ENTERING BLOCK DECLARATION')
  blocks[block_element]=Object(block_element, Block)

problem.add_objects(blocks.values())

#sanity-check 
for obj in problem.all_objects:
  #print('ENTERING PRINT OBJECTS')
  print(obj)

# Declare fluents (states)
on = Fluent("on", BoolType(), below=Block, above=Block)
clear = Fluent("clear", BoolType(), block=Block)
holding = Fluent("holding", BoolType(), block=Block)
hand_empty = Fluent("hand_empty", BoolType())

problem.add_fluents([on, clear, holding, hand_empty])

# Declare actions
pick_up = InstantaneousAction("pick_up", block=Block)
block = pick_up.parameter("block")
pick_up.add_precondition(clear(block))
pick_up.add_precondition(hand_empty())
pick_up.add_effect(holding(block), True)
pick_up.add_effect(hand_empty(), False)
pick_up.add_effect(clear(block), False)
problem.add_action(pick_up)

put_down = InstantaneousAction("put_down", block=Block)
block = put_down.parameter("block")
put_down.add_precondition(holding(block))
put_down.add_effect(holding(block), False)
put_down.add_effect(hand_empty(), True)
put_down.add_effect(clear(block), True)
problem.add_action(put_down)

stack = InstantaneousAction("stack", below=Block, above=Block)
below = stack.parameter("below")
above = stack.parameter("above")
stack.add_precondition(holding(above))
stack.add_precondition(clear(below))
stack.add_effect(holding(above), False)
stack.add_effect(clear(below), False)
stack.add_effect(on(below, above), True)
stack.add_effect(hand_empty(), True)
problem.add_action(stack)

unstack = InstantaneousAction("unstack", below=Block, above=Block)
below = unstack.parameter("below")
above = unstack.parameter("above")
unstack.add_precondition(on(below, above))
unstack.add_precondition(clear(above))
unstack.add_effect(on(below, above), False)
unstack.add_effect(holding(above), True)
unstack.add_effect(clear(below), True)
unstack.add_effect(hand_empty(), False)
problem.add_action(unstack)


def set_clear(block):
  problem.set_initial_value(clear(blocks[block]), True)

def set_hand(boolean):
  problem.set_initial_value(hand_empty(), boolean)

def set_on(block1, block2):
  problem.set_initial_value(on(blocks[block1], blocks[block2]), True)

def set_on_goal(block1,block2):
   problem.add_goal(on(blocks[block1], blocks[block2]))

print('\n PRINTING PROBLEM KIND \n',problem.kind)

with OneshotPlanner(problem_kind=problem.kind) as planner:
    result = planner.solve(problem)
    print("##########%s returned: %s############" % (planner.name, result.plan))
plan = result.plan
print('########%s#######' % plan.kind)

#unified_planning.shortcuts.get_all_applicable_engines(problem_kind=problem.kind)

validator = PlanValidator(name='aries-val')

if validator.validate(problem, plan):
    print('The plan is valid')
else:
    print('The plan is invalid')