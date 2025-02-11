def get_generation_prompt(example_init,example_goal,example_plan,
                          problem_init,problem_goal):
    generation_prompt=('''I am playing with a set of blocks where I need to arrange the blocks into stacks
    Here are the actions I can do: Pick up a block, Unstack a block from on top of another block, Put down a block, Stack a block on top of another block."
    I have the following restrictions on my actions:
    I can only pick up or unstack one block at a time
    I can only pick up or unstack a block if my hand is empty
    I can only pick up a block if the block is on the table and the block is clear
    A block is clear if the block has no other blocks on top of it and if the block is not picked up
    I can only unstack a block from on top of another block if the block I am unstacking was really on top of the other block
    I can only unstack a block from on top of another block if the block I am unstacking is clear
    Once I pick up or unstack a block, I am holding the block
    I can only put down a block that I am holding
    I can only stack a block on top of another block if I am holding the block being stacked
    I can only stack a block on top of another block if the block onto which I am stacking the block is clear
    Once I put down or stack a block, my hand becomes empty
    Once you stack a block on top of a second block, the second block is no longer clear

    [STATEMENT]'''
    f"As initial conditions I have that: {example_init}"
    f"My goal is to have that: {example_goal}" 
    "My plan is as follows:"

    f"[PLAN]{example_plan}[PLAN END]"

    "[STATEMENT]"
    f"As initial conditions I have that: {problem_init}" 
    f"My goal is to have that: {problem_goal}"
    '''My plan is as follows:

    [PLAN]''')
    return generation_prompt

def get_nextaction_prompt(example_init,example_goal,
                          example_action_hist,example_next_action,
                          problem_init,problem_goal,
                          problem_action_hist):
    nextaction_prompt=('''I am playing with a set of blocks where I need to arrange the blocks into stacks
    Here are the actions I can do: Pick up a block, Unstack a block from on top of another block, Put down a block, Stack a block on top of another block.
    I have the following restrictions on my actions:
    I can only pick up or unstack one block at a time
    I can only pick up or unstack a block if my hand is empty
    I can only pick up a block if the block is on the table and the block is clear
    A block is clear if the block has no other blocks on top of it and if the block is not picked up
    I can only unstack a block from on top of another block if the block I am unstacking was really on top of the other block
    I can only unstack a block from on top of another block if the block I am unstacking is clear
    Once I pick up or unstack a block, I am holding the block
    I can only put down a block that I am holding
    I can only stack a block on top of another block if I am holding the block being stacked
    I can only stack a block on top of another block if the block onto which I am stacking the block is clear
    Once I put down or stack a block, my hand becomes empty
    Once you stack a block on top of a second block, the second block is no longer clear

    [STATEMENT]'''
    f"\nAs initial conditions I have that: {example_init}"
    f"\nMy goal is to have that: {example_goal}"
    "\nI work towards the goal state one action at a time:"

    f"\n\n[ACTION HISTORY]\n{example_action_hist}\n[END ACTION HISTORY]\n"

    f"\n[NEXT ACTION] {example_next_action} [END NEXT ACTION]\n\n"

    "[STATEMENT]\n"
    f"As initial conditions I have that: {problem_init}"
    f"\nMy goal is to have that: {problem_goal}"
    "\nI work towards the goal state one action at a time:"

    f"\n\nACTION HISTORY]\n{problem_action_hist}\n[END ACTION HISTORY]\n\n"

    "[NEXT ACTION]")
    return nextaction_prompt

# gen_example_init = "the red block is clear, the orange block is clear, the yellow block is clear, the hand is empty, the orange block is on top of the blue block, the red block is on the table, the blue block is on the table, the yellow block is on the table."
# gen_example_goal = "the red block is on top of the orange block, the yellow block is on top of the red block, blue block is on the table."
# example_plan = '''unstack the orange block from on top of the blue block
# put down the orange block
# pick up the red block
# stack the red block on top of the orange block
# pick up the yellow block
# stack the yellow block on top of the red block'''

# gen_problem_init = "the blue block is clear, the yellow block is clear, the hand is empty, the blue block is on top of the orange block, the orange block is on top of the red block, the red block is on the table, the yellow block is on the table."
# gen_problem_goal = "the red block is on top of the orange block, the blue block is on top of the yellow block."

nextaction_example_init = "the red block is clear, the orange block is clear, the yellow block is clear, the hand is empty, the orange block is on top of the blue block, the red block is on the table, the blue block is on the table, the yellow block is on the table."
nextaction_example_goal = "the red block is on top of the blue block and the yellow block is on top of the red block , the orange block is on the table."
nextaction_example_action_hist = '''unstack the orange block from on top of the blue block
put down the orange block
pick up the red block'''
nextaction_example_nextaction= "stack the red block on top of the blue block"

nextaction_problem_init = "the blue block is clear, the yellow block is clear, the hand is empty, the blue block is on top of the orange block, the orange block is on top of the red block, the red block is on the table, the yellow block is on the table"
nextaction_problem_goal = "the red block is on top of the orange block , the blue block is on top of the yellow block"
