import environment as env
import utils

import numpy as np
from random import choice

NUM_OF_EPISODES = int(1e6)
PRINT_EVERY = int(2e5)
N_ZERO = 10

# Q represents the Q-values: maps (state, action) -> value
# State is bidimensioal with dimensions 10 by 21.
# Action is a either 0 or 1.
Q = np.zeros([11, 22, 2])

# N represents how many times a state-action pair has been visited:
# maps (state, action) -> count
N = np.zeros([11, 22, 2])

def epsilon(state):
  if not (1 <= state[1] <= 21):
    return 0 # Player has busted, that means that this will not matter anyway.
  # Epsilon depends on how many times you visited the state.
  return N_ZERO / (N_ZERO + np.sum(N[state[0], state[1], action] for action in [0, 1]))

def get_Q(state, action):
  if not (1 <= state[1] <= 21):
    return -1 # Player has busted.
  return Q[state[0], state[1], action]

def epsilon_greedy_action(state):
  if np.random.random() > epsilon(state):
    # Greedy move. Expoitation.
    return np.argmax([get_Q(state, action) for action in [0, 1]])
  else:
    # Random move. Exploration.
    return choice([0, 1])

def main():
  for episode in range(NUM_OF_EPISODES):
    game = env.Easy21()
    state = game.get_state()
    action = epsilon_greedy_action(state)
    is_terminal = False

    while not is_terminal:
      new_state, reward, is_terminal = game.step(action)
      new_action = epsilon_greedy_action(new_state)

      # Update the visit count.
      N[state[0], state[1], action] += 1

      # Update the Q value.

      # How much we expect the our current Q-value to be wrong. We estimate the true Q
      # value by looking at the obtained reward and the next Q value we would pick. 
      error = (reward + get_Q(new_state, new_action)) - get_Q(state, action)
      
      # Step-size for the update. Decreases over time, when we visit the state more and 
      # more we are more confident of our value, so we update it less. The fact that alpha
      # will eventually go to zero is essential for convergence of the algorithm toward
      # the optimal policy.
      alpha = 1 / N[state[0], state[1], action]
      
      # Update the Q values using Sarsa algorithm.
      # Q(S, A) <-- Q(S, A) + alpha*( reward + discount*Q(S', A') - Q(S, A) )
      # Note that discount is 1 in this case.
      Q[state[0], state[1], action] = get_Q(state, action) + alpha * error

      # Move to the new state-action pair.
      state = new_state
      action = new_action
    
    if episode % PRINT_EVERY == 0:
      print(episode)
      #print(Q)
      #utils.plot(Q)
  
  utils.plot(Q)


if __name__ == "__main__":
  main()