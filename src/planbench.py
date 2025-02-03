#------------------------------------ PLANBENCH PARSING FUNCTIONS-------------------------------

def get_blocks(predicate):
  blocks=[]
  for idx,word in enumerate(predicate):
      predicate[idx]=word.lower().strip(".,!? ()")
      if word.lower().strip(".,!? ")=="block":
          blocks.append(predicate[idx-1])
  return blocks


def parse_planbench_initial_condition(problem, ic):
  #the blue block is clear, the yellow block is clear, the hand is empty, the blue block is on top of the orange block, the orange block is on top of the red block, the red block is on the table, the yellow block is on the table.

    statements = ic.split(', ')
    for s in statements:
        words=s.split(' ')
        blocks=get_blocks(words)
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

    statements = gs.split(', ')
    for s in statements:
        words=s.split(' ')
        blocks=get_blocks(words)
        
        #make calls to unified planning and define goal state
        if 'top' in words:
        #print('\TOP was called\n')
            problem.set_on_goal(blocks[0],blocks[1])