import numpy as np

N0 = 10

class MontecarloAgent:
  def __init__(self):
    # Q-values: map from the tuple state-action to the Q value:
    # maps (state, action) -> value
    self.Q = {}
    # N represents how many times a state-action pair has been visited:
    # maps (state, action) -> count
    self.N = {}

  def get_key(self, state, action):
    return  tuple([tuple(state), action])

  def get_Q(self, state, action):
    key = self.get_key(state, action)
    if key not in self.Q:
      return 0
    else:
      return self.Q[key]

  def get_N(self, state, action):
    key = self.get_key(state, action)
    if key not in self.N:
      return 0
    else:
      return self.N[key]

  def epsilon(self, state):
    return N0 / (N0 + np.sum([self.get_N(state, action) for action in range(0,9)] ))
  
  def possible_actions(self, state):
    actions = []
    for pos, val in enumerate(state):
      if val == 0:
        actions.append(pos)
    return actions

  def epsilon_greedy_action(self, state):
    if np.random.random() > self.epsilon(state):
      # Greedy move. Expoitation.
      actions = self.possible_actions(state)
      best_action = actions[0]
      best_value = self.get_Q(state, actions[0])
      for action in actions[1:]:
        best_action = action if self.get_Q(state, action) > best_value else best_action
      return best_action
    else:
      # Random move. Exploration.
      return np.random.choice(self.possible_actions(state))
    
  def act(self, state):
    action = self.epsilon_greedy_action(state)
    
    # Increment the count for the state-action pair.
    key = self.get_key(state, action)
    if key in self.N:
      self.N[key] += 1
    else:
      self.N[key] = 1
    return action
  
  def learn(self, states_actions, game_reward):
    for state, action in states_actions:
      alpha = 1 / self.get_N(state, action)
      error = game_reward - self.get_Q(state, action)

      key = self.get_key(state, action)
      if key in self.Q:
        self.Q[key] += alpha * error
      else:
        self.Q[key] = alpha * error
