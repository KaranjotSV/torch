import pdb
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from utils import device, accuracy
from networks import NN, CNN


arch = "CNN"
# pdb.set_trace()
size = 784
channels = 1
classes = 10
rate = 0.001
batch = 64
epochs = 1

# data
train = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train, batch_size=batch, shuffle=True)

test = datasets.MNIST(root='dataset/', train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=test, batch_size=batch, shuffle=True)

# model = NN(input_size=size, classes=classes).to(device)
model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=rate)

# train network
for epoch in range(epochs):
    for batch_ind, (data, targets) in enumerate(train_loader):
        # data to device
        data = data.to(device)
        targets = targets.to(device)

        if arch == "NN":
            # reshape
            data = data.reshape(data.shape[0], -1)
        # forward
        preds = model(data)
        loss = criterion(preds, targets)
        # backward
        optimizer.zero_grad()  # to set all gradients to 0 for each batch
        loss.backward()
        # gradient descent
        optimizer.step()  # updation of weights based upon gradients calculated in loss.backward()

# check accuracy
accuracy(train_loader, model, arch)
accuracy(test_loader, model, arch)
