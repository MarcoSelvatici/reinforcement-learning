from random import randint
from enum import Enum

class Color(Enum):
  RED = 0
  BLACK = 1

# The state of the game is defined by the tuple (dealer_first_card, player_sum).
# The action is either HIT or STICK
class Easy21:
  def __init__(self):
    self.dealer_first_card = randint(1, 10)
    self.player_sum = randint(1, 10)

  def get_state(self):
    return (self.dealer_first_card, self.player_sum)
  
  def draw_card(self):
    # Card is red with probability 1/3, and black with probability 2/3.
    return (Color.RED if (randint(0,2) == 2) else Color.BLACK, randint(1, 10))
  
  def run_dealer_policy(self):
    cur_sum = self.dealer_first_card
    while cur_sum >= 1 and cur_sum < 17:
      card_color, card_value = self.draw_card()
      if card_color == Color.BLACK:
        cur_sum += card_value
      else:
        cur_sum -= card_value
    return cur_sum
  
  def is_bust(self, sum):
    return sum < 1 or sum > 21

  def step(self, action):
    reward = 0
    is_terminal = False

    if action == 1:
      card_color, card_value = self.draw_card()
      if card_color == Color.BLACK:
        self.player_sum += card_value
      else:
        self.player_sum -= card_value

      if self.is_bust(self.player_sum):
        reward = -1
        is_terminal = True

    else:
      final_dealer_value = self.run_dealer_policy()
      is_terminal = True
      if self.is_bust(final_dealer_value) or self.player_sum > final_dealer_value:
        reward = +1
      elif final_dealer_value == self.player_sum:
        reward = 0
      else:
        reward = -1

    return ((self.dealer_first_card, self.player_sum), reward, is_terminal)
