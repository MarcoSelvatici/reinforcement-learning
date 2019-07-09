from environment import TicTacToe
from montecarlo_agent import MontecarloAgent
from random_agent import RandomAgent
from human_agent import HumanAgent

from copy import deepcopy

NUM_OF_TRAINING_GAMES = int(2e5)
BATCH_SIZE = int(1e4)

NUM_OF_DISPLAY_GAMES = 2

win_p1 = 0
draw = 0

p1 = MontecarloAgent()
#p2 = MontecarloAgent()
p2 = RandomAgent()

# Training.
for game_num in range(NUM_OF_TRAINING_GAMES):
  game = TicTacToe()

  states_actions_p1 = []
  #states_actions_p2 = []

  is_terminal = False
  game_reward_p1 = 0
  game_reward_p2 = 0

  # p1 starts.
  turn = 1
  state = game.get_state()

  while not is_terminal:
    if turn == 1:
      action = p1.act(state)
      states_actions_p1.append((deepcopy(state), action))
      state, game_reward_p1, is_terminal = \
        game.step(player=1, action=action, display=False)
      game_reward_p2 = game_reward_p1 * -1
      turn = 2
    elif turn == 2:
      action = p2.act(state)
      # states_actions_p2.append((deepcopy(state), action))
      state, game_reward_p2, is_terminal = \
        game.step(player=2, action=action, display=False)
      game_reward_p1 = game_reward_p2 * -1
      turn = 1
  
  p1.learn(states_actions_p1, game_reward_p1)
  #p2.learn(states_actions_p2, game_reward_p2)

  if game_reward_p1 == 1:
    win_p1 += 1
  elif game_reward_p1 == 0:
    draw += 1
  
  if game_num % BATCH_SIZE == 0:
    print("Batch: {}".format(game_num / BATCH_SIZE))
    print("P1:   {}\nDRAW: {}\nP2:   {}".format(win_p1, draw, BATCH_SIZE - win_p1 - draw))
    print()
    win_p1 = 0
    draw = 0

p2 = HumanAgent()

# Games to display what has been learnt.
for game_num in range(NUM_OF_DISPLAY_GAMES):
  print("\n## Game ##")

  game = TicTacToe()

  is_terminal = False
  game_reward_p1 = 0
  game_reward_p2 = 0

  # p1 starts.
  turn = 1
  state = game.get_state()

  while not is_terminal:
    if turn == 1:
      action = p1.act(state)
      state, game_reward_p1, is_terminal = \
        game.step(player=1, action=action, display=True)
      game_reward_p2 = game_reward_p1 * -1
      turn = 2
    elif turn == 2:
      action = p2.act(state)
      state, game_reward_p2, is_terminal = \
        game.step(player=2, action=action, display=True)
      game_reward_p1 = game_reward_p2 * -1
      turn = 1
