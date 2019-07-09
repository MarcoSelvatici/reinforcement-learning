import numpy as np

class HumanAgent:
  def possible_actions(self, state):
    actions = []
    for pos, val in enumerate(state):
      if val == 0:
        actions.append(pos)
    return actions
    
  def act(self, state):
    action = -1
    actions = self.possible_actions(state)
    while action not in actions:
      action = int(input("Possible actions: {}".format(actions)))
    return action
