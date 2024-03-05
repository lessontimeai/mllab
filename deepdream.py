import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Step 1: Load and normalize the MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# Step 2: Define a Multilayer Perceptron
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(28*28, 128) # Flatten the 28x28 images into vectors of size 784
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10) # Output layer, 10 classes

    def forward(self, x):
        x = x.view(-1, 28*28) # Flatten the images
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x) # No activation, raw output for CrossEntropyLoss
        return x

net = MLP()

# Step 3: Define a Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

# Step 4: Train the network
for epoch in range(10):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 100 == 99:    # print every 100 mini-batches
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0

print('Finished Training')

# Step 5: Test the network on the test data
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))


class DeepDream(nn.Module):
    def __init__(self, mlp):
        super(DeepDream, self).__init__()
        self.embedding = nn.Embedding(1, 28*28)
        self.embedding.weight = nn.Parameter(inputs[0].reshape(1,28*28))
        self.mlp = mlp

    def forward(self, idx):
        e = self.embedding(idx)
        o = self.mlp(e)
        return e, o

deepdream = DeepDream(net)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(deepdream.embedding.parameters(), lr=0.01, momentum=0.09)    
for rep in range(1000):
    optimizer.zero_grad()
    embeddings, outputs = deepdream(torch.zeros(1,1).long())
    loss = criterion(outputs, torch.tensor([1]))
    loss.backward()
    optimizer.step()
    print(loss.item())