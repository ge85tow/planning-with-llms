# -*- coding: utf-8 -*-
from unified_planning.shortcuts import *
from unified_planning.plans import *
import sys
from unified_planning.engines import sequential_simulator

import sys
sys.path.append("/srv/chawak/planning-with-llms/src/shared")
import llm_utils


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

        on_table=Fluent('on_table',BoolType(),block=Block)
        self.on_table=on_table
        
        holding = Fluent('holding', BoolType(), block=Block)
        self.holding = holding

        hand_empty = Fluent('hand_empty', BoolType())
        self.hand_empty = hand_empty
        
        self.add_fluent(on,default_initial_value=False)
        self.add_fluent(clear,default_initial_value=False)
        self.add_fluent(on_table,default_initial_value=False) 
        self.add_fluent(holding,default_initial_value=False)
        self.add_fluent(hand_empty,default_initial_value=False)
        
        
        # Declare actions
        pick_up = InstantaneousAction('pick_up', block=Block)
        block = pick_up.parameter('block')
        #pre-req
        pick_up.add_precondition(clear(block))
        pick_up.add_precondition(hand_empty())
        pick_up.add_precondition(on_table(block))
        #effects
        pick_up.add_effect(on_table(block),False)
        pick_up.add_effect(holding(block),True)
        pick_up.add_effect(hand_empty(), False)
        pick_up.add_effect(clear(block), False)
        self.add_action(pick_up)
        self.pick_up = pick_up

        put_down = InstantaneousAction('put_down', block=Block)
        block = put_down.parameter('block')
        #pre-req
        put_down.add_precondition(holding(block))
        #effects
        put_down.add_effect(holding(block), False)
        put_down.add_effect(hand_empty(),True)
        put_down.add_effect(clear(block),True)
        put_down.add_effect(on_table(block),True)
        self.add_action(put_down)
        self.put_down = put_down
        
        stack = InstantaneousAction('stack', above=Block, below=Block)
        below = stack.parameter('below')
        above = stack.parameter('above')
        #pre-req
        stack.add_precondition(holding(above))
        stack.add_precondition(clear(below))
        #effects
        stack.add_effect(holding(above), False)
        stack.add_effect(clear(below), False)
        stack.add_effect(clear(above),True)
        stack.add_effect(on(above, below),True)
        stack.add_effect(hand_empty(),True)
        self.add_action(stack)
        self.stack = stack
        
        unstack = InstantaneousAction('unstack', above=Block, below=Block)
        below = unstack.parameter('below')
        above = unstack.parameter('above')
        #pre-req
        unstack.add_precondition(on(above, below))
        unstack.add_precondition(clear(above))
        unstack.add_precondition(hand_empty())
        #effects
        unstack.add_effect(on(above, below), False)
        unstack.add_effect(holding(above),True)
        unstack.add_effect(clear(below),True)
        unstack.add_effect(hand_empty(), False)
        unstack.add_effect(clear(above), False)
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

    def set_on_table(self,block:str):
        self.set_initial_value(self.on_table(self.blocks[block]),True)

    def set_hand(self, boolean):
        self.set_initial_value(self.hand_empty, boolean)

    def set_on_goal(self, block1: str, block2:str ):
        self.add_goal(self.on(self.blocks[block1], self.blocks[block2]))
    
    def set_clear_goal(self, block1: str):
        self.add_goal(self.clear(self.blocks[block1]))

    def set_ontable_goal(self,block1:str):
        self.add_goal(self.on_table(self.blocks[block1]))
    
    def create_seq_simulation(self):
        return SequentialSimulator(self,name='sequential_simulator')
        
    def generate_plan(self):
        with OneshotPlanner(name='pyperplan') as planner:
            result = planner.solve(self)
            #print('##########%s returned: %s############' % (planner.name, result.plan))
            plan = result.plan
            #print('########%s#######' % plan.kind)
        return result


    def create_action(self, action, blocks):
        #print(f'BLOCKS IN ACTION : {blocks}')
        
        #check if solution blocks matches problem blocks
        for block in blocks:
            if block not in self.blocks:
                print('\n !!! INCORRECT objects being acted on')
                return False


        if action == 'stack':
            return ActionInstance(self.stack, (self.blocks[blocks[0]], self.blocks[blocks[1]]))
        if action == 'unstack':
            return ActionInstance(self.unstack, (self.blocks[blocks[0]], self.blocks[blocks[1]]))
        if action == 'pick-up':
            return ActionInstance(self.pick_up, (self.blocks[blocks[0]]))
        if action == 'put-down':
            return ActionInstance(self.put_down, (self.blocks[blocks[0]]))
        else:
            return False

    def create_plan_from_tuples(self, action_tuples):
        actions = []
        for a in action_tuples:
            if not a:
                return False
            
            predicate = a[0]
            blocks = a[1:]
            action=self.create_action(predicate, blocks)
            if action: 
                actions.append(action)
            else: 
                return False

        model_plan = SequentialPlan(actions = actions)
        return model_plan
    
    def GRPOcreate_plan_from_tuples(self, action_tuples):
        actions = []
        
        for a in action_tuples: 
            if not a:
                break  #break @ invalid action
            predicate = a[0]
            blocks = a[1:]
            action=self.create_action(predicate, blocks)
            if action: 
                actions.append(action)
            else: 
                break

        if actions:
            model_plan = SequentialPlan(actions = actions)
            return model_plan
        else:
            print("No VALID ACTION found")
            return False
    
    def make_plan(self,actions):
        model_plan=None
        model_plan = SequentialPlan(actions=actions)
        return model_plan
    
    
    def validate_action(self, state, action, simulation):
        if not action:
            return False
        check=simulation.is_applicable(state,action)
        if check:
            new_state=simulation.apply(state,action)
            return new_state
        else:
            return False

    def check_and_apply(self,sim,model_plan):
        
        flag=True
        current_state=sim.get_initial_state()
        counter=0
        if model_plan:
            # print(f'\n\nEntering "CHECK & APPLY"')
            for next_action in model_plan.actions:
                # print(f'\n Valid actions so far : {counter}')
                # print(f'\n Current State : {current_state}')
                new_state=self.validate_action(current_state,next_action,sim)
                # print(f'\n Next Action : {next_action}')
                #ACTION is invalid
                if not new_state:
                    print(f"\n\n!! INVALID ACTION SEQUENCE:{next_action}")
                    flag=False
                    break
                else:
                    counter+=1
                    current_state=new_state
                    # print(f'\n New State : {new_state}')
                print('\n','-'*20)   
        #print(f'\n Returning valid-action-count: {counter}')
        return current_state,flag,counter
    
    def GRPO_check_and_apply(self,sim,model_plan):
        
        distance2goal=[]
        current_state=sim.get_initial_state()
        counter=0
        if model_plan:
            print('\n','-'*60) 
            print(f'\n\nEntering "CHECK & APPLY"')
            for next_action in model_plan.actions:
                print(f'\n Valid actions so far : {counter}')
                print(f'\n Current State : {current_state}')

                #get distance to goal before each valid action is applied, starting from init
                distance=self.actions_to_goal(current_state)
                print(f"\n Distance to goal from current state is {distance}")
                distance2goal.append(distance)

                new_state=self.validate_action(current_state,next_action,sim)
                
                print(f'\n Next Action : {next_action}')
                
                #if ACTION is invalid
                if not new_state:
                    print(f"\n\n!! INVALID ACTION SEQUENCE:{next_action}")
                    break
                else:
                    counter+=1
                    current_state=new_state
                    print(f'\n New State : {new_state}')
                
                print('\n','-'*60)   
        #print(f'\n Returning valid-action-count: {counter}')
        return current_state,counter,distance2goal

    def get_sim_for_state(self, state):
        sim=self.clone()
        state_fluents=list(state._values)
        state_values=[state.get_value(element).constant_value() for element in state_fluents]

        for fluent,value in zip(state_fluents,state_values):
            sim.set_initial_value(fluent,value)   
        return sim

#makes a simulation with current state and solves it to compute actions to goal
    def actions_to_goal(self,state):

        print(f'\n Computing "Actions To Goal"')
        sim=self.get_sim_for_state(state) #get simulation with last valid state as initial state
        res=None
        with OneshotPlanner(problem_kind=sim.kind) as planner:
            res = planner.solve(sim)
            
        plan = res.plan
        plan_actions=str(plan).strip().split('\n')
        plan_actions=plan_actions[1:]
        steps=len(plan_actions)
        return steps


    def terminate(self,sim_state):

        #PROBLEM: goals
        list_goals=[str(goal) for goal in self.goals]

        #SIMULATION: STATE
        state_values=list(sim_state._values)

        #extract true state pairs from all states
        true_states=[]
        for element in state_values:
            value=sim_state.get_value(element).constant_value()
            if value==True:
                true_states.append(str(element))

        #check for similarity      
        goals_met=set(list_goals).issubset(true_states)

        print(f'GOALS: {set(list_goals)}')
        print(f'Current state: {true_states}')
        print(f'??????????????Goals met:{goals_met}')
        return goals_met
        
      
    def validate_plan(self, plan):
        with PlanValidator(name="aries-val") as validator:
            result = validator.validate(self.clone(), plan)
        return result
