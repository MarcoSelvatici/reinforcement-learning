import gym
import torch

from time import sleep
from random import randint

from agent_memory import Memory
from simple_NN import NN

# Create game environment.
# MountainCar-v0
# CartPole-v1
env = gym.make('CartPole-v1')

# Create DQN agent.
state_size = 4 # Same size as observation space.
hidden_size = 16 # Arbitrary
number_actions = env.action_space.n

gamma = 1.0 # Time discount.
epsilon = 0.01

Q_agent = NN(state_size, hidden_size, number_actions, learning_rate=0.00005)

# Create events memory.
memory = Memory()

# Other hyper parameters.
number_of_training_games = int(1e3)
number_of_test_games = int(10)

batch_avg = 0

for game in range(number_of_training_games):
  state = env.reset()
  done = False
  score = 0
  while not done:
    # Extract which action is predicted with the best value, or act randomly with
    # probability epsilon.
    _, action = torch.max(Q_agent.forward(torch.tensor(state).double()), 0)
    action = action.item()
    if torch.rand(1).item() < epsilon:
      action = randint(0, number_actions - 1)
    
    # Perform the action.
    new_state, reward, done, _ = env.step(action)
    score += reward

    # Save to samples memory.
    memory.push(state, action, new_state, reward, done)

    # Get a bunch of samples from the memory and use them to train the network.
    samples = memory.get_samples(batch_size=8)
    ideal_values = []
    predicted_values = []
    
    for sample in samples:
      if sample is None:
        continue
      b_state, b_action, b_new_state, b_reward, b_done = sample
      
      # Estimate of the ideal policy from b_state (that is the reward, plus the
      # discounted best possible action from next state).
      ideal_values.append(gamma * \
                          torch.max(Q_agent.forward(torch.tensor(b_new_state).double())) \
                          + b_reward if not b_done else b_reward)
      # Predicted value for the pair (b_state, b_action) from our NN.
      predicted_values.append(Q_agent.forward(torch.tensor(b_state).double())[b_action])
    
    if len(ideal_values) > 0:
      Q_agent.fit(torch.tensor(predicted_values).double(),\
                  torch.tensor(ideal_values).double())
    state = new_state
  
  batch_avg += score

  if game % 100 == 0:
    print("game: {}  avg_score: {}".format(game, batch_avg / 100))
    batch_avg = 0

input("Training is over. Press any key to see sample games...")

for game in range(number_of_test_games):
  state = env.reset()
  done = False
  while not done:
    env.render()
    # Extract which action is predicted with the best value.
    _, action = torch.max(Q_agent.forward(torch.tensor(state).double()), 0)
    # Perform the action.
    new_state, reward, done, _ = env.step(action.item())
    sleep(0.02)
    state = new_state

env.close()
