import numpy as np

class RandomAgent:
  def possible_actions(self, state):
    actions = []
    for pos, val in enumerate(state):
      if val == 0:
        actions.append(pos)
    return actions
    
  def act(self, state):
    return np.random.choice(self.possible_actions(state))
