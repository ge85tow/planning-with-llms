import re

#------------------------------------ PLANBENCH PARSING FUNCTIONS-------------------------------

def get_blocks(predicate):
  blocks=[]
  for idx,word in enumerate(predicate):
      predicate[idx]=word.lower().strip(".,!? ()")
      if word.lower().strip(".,!? ")=="block":
          blocks.append(predicate[idx-1])
  return blocks


def parse_problem(pbstring):
    matches = [m.start() for m in re.finditer(r'\[STATEMENT\]', pbstring)]
    if len(matches) >= 2:
        problem_string = pbstring[matches[1]:]
        
    conditions=problem_string.split("As initial conditions I have that,")[1]
    init=conditions.split('.')[0]
    
    goal_part=conditions.split("My goal is to have that")[1]
    goal=goal_part.split('.')[0]
    
    return init,goal

def parse_planbench_initial_condition(problem, ic):
  #the blue block is clear, the yellow block is clear, the hand is empty, the blue block is on top of the orange block, the orange block is on top of the red block, the red block is on the table, the yellow block is on the table.
    #print('I HAVE ENTERED PARSING INITIAL CONDITION')
    #ic = re.sub(r' and ', ', ', ic)
    statements = ic.split(', ')
    for s in statements:
        words=s.split(' ')
        blocks=get_blocks(words)
        #print(f'Blocks being handled:{blocks}')
        for block in blocks:
            if block not in problem.blocks:
                problem.add_blocks(block)
        #print('IC from Gen Prompt small loop %s' % (words))
        #print('blocks from Gen Prompt small loop',blocks)
        
        #make calls to unified planning and define initial conditions
        if 'clear' in words:
            #print('\nCLEAR was called\n')
            problem.set_clear(blocks[0])
        
        if 'top' in words:
            #print('\TOP was called\n')
            problem.set_on(blocks[0],blocks[1])
        
        if 'hand' in words:
            #print('\HAND was called\n')
            problem.set_hand(True)

def parse_planbench_goal_state(problem, gs):
#My goal is to have that: the red block is on top of the orange block, the blue block is on top of the yellow block.
    #gs=gs.replace('and',',')
    #print('I HAVE ENTERED PARSING GOAL')
    statements = gs.split(',')
    for s in statements:
        words=s.split(' ')
        blocks=get_blocks(words)
        
        #make calls to unified planning and define goal state
        if 'top' in words:
            #print('\TOP was called\n')
            problem.set_on_goal(blocks[0],blocks[1])
        
        if 'clear' in words:
            #print('\CLEAR was called\n')
            problem.set_clear_goal(blocks[0])