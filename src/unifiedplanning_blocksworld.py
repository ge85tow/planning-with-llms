# -*- coding: utf-8 -*-
from unified_planning.shortcuts import *
from unified_planning.plans import *
import llm_utils
from unified_planning.engines import sequential_simulator

class BlocksworldProblem(Problem):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        Block = UserType('Block')
        
        self.Block = Block
        self.blocks = {}
        
        # Declare fluents
        on = Fluent('on', BoolType(), above=Block, below=Block)
        self.on = on
        clear = Fluent('clear', BoolType(), block=Block)
        self.clear = clear
        holding = Fluent('holding', BoolType(), block=Block)
        self.holding = holding
        hand_empty = Fluent('hand_empty', BoolType())
        self.hand_empty = hand_empty
        
        self.add_fluent(on,default_initial_value=False)
        self.add_fluent(clear,default_initial_value=False)
        self.add_fluent(holding,default_initial_value=False)
        self.add_fluent(hand_empty,default_initial_value=False)
        # self.add_fluents([on, clear, holding, hand_empty])
        
        # Declare actions
        pick_up = InstantaneousAction('pick_up', block=Block)
        block = pick_up.parameter('block')
        pick_up.add_precondition(clear(block))
        pick_up.add_precondition(hand_empty())
        pick_up.add_effect(holding(block), True)
        pick_up.add_effect(hand_empty(), False)
        pick_up.add_effect(clear(block), False)
        self.add_action(pick_up)
        self.pick_up = pick_up
        
        put_down = InstantaneousAction('put_down', block=Block)
        block = put_down.parameter('block')
        put_down.add_precondition(holding(block))
        put_down.add_effect(holding(block), False)
        put_down.add_effect(hand_empty(), True)
        put_down.add_effect(clear(block), True)
        self.add_action(put_down)
        self.put_down = put_down
        
        stack = InstantaneousAction('stack', above=Block, below=Block)
        below = stack.parameter('below')
        above = stack.parameter('above')
        stack.add_precondition(holding(above))
        stack.add_precondition(clear(below))
        stack.add_effect(holding(above), False)
        stack.add_effect(clear(below), False)
        stack.add_effect(on(above, below), True)
        stack.add_effect(hand_empty(), True)
        self.add_action(stack)
        self.stack = stack
        
        unstack = InstantaneousAction('unstack', above=Block, below=Block)
        below = unstack.parameter('below')
        above = unstack.parameter('above')
        unstack.add_precondition(on(above, below))
        unstack.add_precondition(clear(above))
        unstack.add_effect(on(above, below), False)
        unstack.add_effect(holding(above), True)
        unstack.add_effect(clear(below), True)
        unstack.add_effect(hand_empty(), False)
        self.add_action(unstack)
        self.unstack = unstack
    
    def get_on(self):
            return self.on
    
    def add_blocks(self, *names: [str]):
        new_blocks = {name: Object(name, self.Block) for name in names 
                      if name not in self.blocks}
        self.blocks |= new_blocks
        self.add_objects(new_blocks.values())

    def set_on(self, block1: str, block2:str):
        self.set_initial_value(self.on(self.blocks[block1],
                                                  self.blocks[block2]),
                                       True)
    def set_clear(self, block: str):
        self.set_initial_value(self.clear(self.blocks[block]), True)

    def set_hand(self, boolean):
        self.set_initial_value(self.hand_empty, boolean)

    def set_on_goal(self, block1: str, block2:str ):
        self.add_goal(self.on(self.blocks[block1], self.blocks[block2]))
    
    def create_seq_simulation(self):
        return SequentialSimulator(self,name='sequential_simulator')
        
    def generate_plan(self):
        with OneshotPlanner(name='pyperplan') as planner:
            result = planner.solve(self)
            print('##########%s returned: %s############' % (planner.name, result.plan))
            plan = result.plan
            print('########%s#######' % plan.kind)
        return result

    def create_action(self, action, blocks):
        print(blocks)
        blocks[0]
        if action == 'stack':
            return ActionInstance(self.stack, (self.blocks[blocks[0]], self.blocks[blocks[1]]))
        if action == 'unstack':
            return ActionInstance(self.unstack, (self.blocks[blocks[0]], self.blocks[blocks[1]]))
        if action == 'pick-up':
            return ActionInstance(self.pick_up, (self.blocks[blocks[0]]))
        if action == 'put-down':
            return ActionInstance(self.put_down, (self.blocks[blocks[0]]))
        else:
            return 'invalid action'

    def create_plan_from_tuples(self, action_tuples):
        actions = []
        for a in action_tuples:
            predicate = a[0]
            blocks = a[1:]
            actions.append(self.create_action(predicate, blocks))
        model_plan = SequentialPlan(actions = actions)
        return model_plan
    
    def validate_action(self, state, action, simulation):
        check=simulation.is_applicable(state,action)
        if check:
            new_state=simulation.apply(state,action)
            return new_state
        else:
            return False

    def check_and_apply(self,sim,model_plan):
        
        flag=True
        current_state=sim.get_initial_state()
        for next_action in model_plan.actions:
            new_state=self.validate_action(current_state,next_action,sim)
            
            #ACTION is invalid
            if not new_state:
                print(f"\n\n!! INVALID ACTION SEQUENCE:{next_action}")
                flag=False
                break
            else:
                current_state=new_state
        
        return sim if flag else False

    def terminate(self,sim,sim_state):

        #PROBLEM: goals
        list_goals=[str(goal) for goal in self.goals]

        #SIMULATION: STATE
        state_values=list(sim_state._values)
        state_value_set = {str(element) for element in state_values}  

        #check for similarity      
        goals_met=set(list_goals).issubset(state_value_set)

        if not goals_met:
            print(f"All problem goals have not been met")
            return False
        else:
            return True
      


    def validate_plan(self, plan):
        with PlanValidator(name="aries-val") as validator:
            result = validator.validate(self.clone(), plan)
        return result

