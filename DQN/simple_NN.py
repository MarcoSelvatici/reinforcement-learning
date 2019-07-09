# Class that implements a simple NN with one hidden layer.

import torch

# For MINST dataset.
import torchvision
import torchvision.transforms as transforms

DEFAULT_LEARNING_RATE = 1e-4

class NNModel(torch.nn.Module):
  def __init__(self, input_size, hidden_size, output_size):
    super(NNModel, self).__init__()
    
    # Define the layers.
    # fc1  -> first set of weights.
    # relu -> activation function of the hidden layer.
    # fc2  -> second set of weights.
    self.fc1 = torch.nn.Linear(input_size, hidden_size)
    self.relu = torch.nn.ReLU()
    self.fc2 = torch.nn.Linear(hidden_size, output_size)
  
  def forward(self, x):
    y = self.fc1(x)
    y = self.relu(y)
    y = self.fc2(y)
    return y

class NN():
  def __init__(self, input_size, hidden_size, output_size,
               learning_rate=DEFAULT_LEARNING_RATE):
    self.model = NNModel(input_size, hidden_size, output_size).double()
    # Define optimizer and loss function.
    self.loss_fn = torch.nn.MSELoss()
    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
  
  def forward(self, x):
    return self.model.forward(x)
  
  def fit(self, predicted, actual):
    self.optimizer.zero_grad()
    loss = self.loss_fn(predicted, actual)
    loss = torch.autograd.Variable(loss, requires_grad = True)
    loss.backward()
    self.optimizer.step()

    return loss.item()
  

def test():
  # Test with MINST dataset.
  # A typical accurancy after 5 epochs is 

  # Hyper-parameters.
  input_size = 784
  hidden_size = 500
  output_size = 10
  num_epochs = 5
  batch_size = 100

  # MNIST dataset.
  train_dataset = torchvision.datasets.MNIST(root='./MNIST_data', 
                                            train=True, 
                                            transform=transforms.ToTensor(),  
                                            download=True)

  test_dataset = torchvision.datasets.MNIST(root='./MNIST_data',
                                            train=False, 
                                            transform=transforms.ToTensor())


  # Data loader.
  train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                            batch_size=batch_size, 
                                            shuffle=True)

  test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                            batch_size=batch_size, 
                                            shuffle=False)

  device = 'cpu'
  net = NN(input_size, hidden_size, output_size)

  # Train the model.
  total_step = len(train_loader)
  for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):  
      # Move tensors to the configured device.
      images = images.reshape(-1, 28*28).to(device)
      labels = labels.to(device)

      # Forward pass.
      outputs = net.forward(images)
      one_hot_labels = []
      for label in labels:
        one_hot_labels.append([1.0 if i == label else 0.0 for i in range(10)])
      
      loss = net.fit(outputs, torch.tensor(one_hot_labels))

      if (i+1) % 100 == 0:
        print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                .format(epoch+1, num_epochs, i+1, total_step, loss))

  # Test the model.
  with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
      images = images.reshape(-1, 28*28).to(device)
      labels = labels.to(device)
      outputs = net.forward(images)
      _, predicted = torch.max(outputs.data, 1)
      total += labels.size(0)
      correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the 10000 test images: {} %'\
          .format(100 * correct / total))
