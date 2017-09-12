import torch
import torchvision
import torchvision.transforms as transforms

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# 参考;
# http://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

# dataset
transform = transforms.ToTensor()
set_train = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)  # [(1x28x28, 1)]
loader_train = torch.utils.data.DataLoader(set_train, batch_size=4, shuffle=True, num_workers=2)
set_test = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
loader_test = torch.utils.data.DataLoader(set_test, batch_size=4, shuffle=True, num_workers=2)


# model

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.lin = nn.Linear(28 * 28, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        y = F.softmax(self.lin(x))
        return y


model = Net()
if torch.cuda.is_available():
    print('running on GPU')
    model.cuda()
else:
    print('running on CPU')
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# training
EPOCHS = 11
VIEW_INTERVAL = 2000
for epoch in range(EPOCHS):
    running_loss = 0
    for i, data in enumerate(loader_train):
        x, y = data
        if torch.cuda.is_available():
            x, y = x.cuda(), y.cuda()
        X, Y = Variable(x), Variable(y)
        optimizer.zero_grad()
        Y_pred = model(X)
        loss = criterion(Y_pred, Y)
        loss.backward()
        optimizer.step()

        # report loss
        running_loss += loss.data[0]
        if i % VIEW_INTERVAL == VIEW_INTERVAL - 1:
            print("Epoch {}, iteration {}; loss: {:.3f}".format(epoch + 1, i + 1, running_loss / VIEW_INTERVAL))
            running_loss = 0

            # testing
            count_correct = 0
            count_total = 0
            for images, labels in loader_test:
                if torch.cuda.is_available():
                    images, labels = images.cuda(), labels.cuda()
                y_pred = model(Variable(images))
                _, labels_pred = torch.max(y_pred.data, 1)
                c = (labels_pred == labels).squeeze()
                for k in c:
                    count_correct += k
                    count_total += 1
            print("  Test Acc: {}".format(count_correct / count_total))
