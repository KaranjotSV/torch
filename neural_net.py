import pdb
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader

import torchvision.datasets as datasets
import torchvision.transforms as transforms


class NN(nn.Module):
    def __init__(self, input_size, classes):
        super(NN, self).__init__()  # super calls the init method of the parent class(nn.Module)
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, classes)

    def forward(self, inp):
        out = F.relu(self.fc1(inp))
        out = self.fc2(out)
        return out


model = NN(784, 10)
data = torch.rand(64, 784)  # 64 samples in a batch

# print(model(data).shape)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# pdb.set_trace()
size = 784
classes = 10
rate = 0.001
batch = 64
epochs = 1

# data
train = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train, batch_size=batch, shuffle=True)

test = datasets.MNIST(root='dataset/', train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=test, batch_size=batch, shuffle=True)

model = NN(input_size=size, classes=classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=rate)

# train network
for epoch in range(epochs):
    for batch_ind, (data, targets) in enumerate(train_loader):
        # data to device
        data = data.to(device)
        targets = targets.to(device)
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
def accuracy(loader, model):
    if loader.dataset.train:
        print("checking on train")
    else:
        print("checking on test")

    correct_count = 0
    total = 0
    model.eval()  # to change working mode of model

    with torch.no_grad():  # to turn off gradients computation
        for data, targets in loader:
            data = data.to(device)
            targets = targets.to(device)
            data = data.reshape(data.shape[0], -1)

            score = model(data)
            _, preds = score.max(dim=1)
            correct_count += (preds == targets).sum()
            total += preds.shape[0]

        print(f"{correct_count}/{total} - accuracy {float(correct_count) / float(total) * 100:.3f}")
    model.train()


accuracy(train_loader, model)
accuracy(test_loader, model)
