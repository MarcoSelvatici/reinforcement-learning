from random import randint

class Memory():
  def __init__(self, size=2048):
    self.size = size
    self.idx = 0
    self.m = [None for _ in range(size)]
  
  def push(self, state, action, state1, reward, done):
    self.m[self.idx] = (state, action, state1, reward, done)
    self.idx = (self.idx + 1) % self.size
  
  def get_samples(self, batch_size=64):
    return [self.m[randint(0, self.size - 1)] for _ in range(batch_size)]
    