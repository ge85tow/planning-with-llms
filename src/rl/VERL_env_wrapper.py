import sys
sys.path.append("/srv/chawak/planning-with-llms/src/")
from shared import unifiedplanning_blocksworld as bw

class BlocksworldEnv:
    def __init__(self, init=None, goal=None):
        self.problem = bw.BlocksworldProblem()
        self.init = init
        self.goal = goal
        self.current_state = None

    def reset(self, init=None, goal=None):
        # Optionally accept new init/goal for each episode
        self.init = init or self.init
        self.goal = goal or self.goal
        # Parse/init state and goal as in your code
        # (Call prompts.parse_init, prompts.parse_goal, etc.)
        # ... (your code here)
        self.current_state = self.problem.initial_values
        return self.current_state

    def step(self, action):
        # action: typically a string or action tuple
        # Apply action and update state
        # e.g., self.problem.apply_action(self.current_state, action)
        # ... (your code here)
        # Compute reward (possibly with your custom functions)
        reward = 0  # compute reward
        done = False  # determine if goal is reached or episode terminates
        info = {}  # any extra info you want
        return self.current_state, reward, done, info

    # Optionally: render(), close(), seed()