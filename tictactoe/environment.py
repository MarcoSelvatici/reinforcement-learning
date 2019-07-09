import sys

# 0 1 2
# 3 4 5
# 6 7 8
WINS = [
  [0, 1, 2], # Horizontal.
  [3, 4, 5],
  [6, 7, 8],
  [0, 3, 6], # Vertical.
  [1, 4, 7],
  [2, 5, 8],
  [0, 4, 8], # Diagonal.
  [2, 4, 6]
]

EMPTY = 0
P1 = 1
P2 = 2

class TicTacToe:
  def __init__(self):
    self.board = [0 for _ in range(9)]
    self.moves = 0
  
  def display_board(self):
    print()
    print(self.board[:3])
    print(self.board[3:6])
    print(self.board[6:9])
    print()

  def get_state(self):
    return self.board
  
  # Returns the winner of the game, or EMPTY if there is no winner.
  def winner(self):
    for win_combination in WINS:
      if (self.board[win_combination[0]] == \
          self.board[win_combination[1]] == \
          self.board[win_combination[2]] != EMPTY):
        return self.board[win_combination[0]]
    return EMPTY
  
  def step(self, player, action, display=False):
    if self.board[action] != EMPTY or not (0 <= action <= 8):
      self.display_board()
      sys.exit("Player {} tried to place in invalid position {}."\
               .format(player, action))
    self.board[action] = player
    self.moves += 1

    if (display):
      self.display_board()
    
    reward = 0
    is_terminal = True if self.moves == 9 else False

    winner = self.winner()
    if winner == player:
      reward = 1
      is_terminal = True
    elif winner != EMPTY:
      reward = -1
      is_terminal = True

    return self.board, reward, is_terminal


# game = TicTacToe()
# is_terminal = False
# turn = 1
# while not is_terminal:
#   action = int(input())
#   _, _, is_terminal = game.step(turn, action, True)
#   if turn == 1:
#     turn = 2
#   elif turn == 2:
#     turn = 1
