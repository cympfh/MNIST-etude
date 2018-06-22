import torch
import torchvision
import torchvision.transforms as transforms

from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim


# dataset
batch_size = 30
transform = transforms.ToTensor()
set_train = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)  # [(1x28x28, 1)]
loader_train = torch.utils.data.DataLoader(set_train, batch_size=batch_size, shuffle=True, num_workers=2)
set_test = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
loader_test = torch.utils.data.DataLoader(set_test, batch_size=batch_size, shuffle=False, num_workers=2)


# model

seq_len = 28
input_dim = 28
hidden_dim = 128
classes_num = 10


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, 1, batch_first=True)
        self.lin = nn.Linear(hidden_dim, classes_num)

    def forward(self, x):

        h0 = torch.zeros(1, x.size(0), hidden_dim)
        c0 = torch.zeros(1, x.size(0), hidden_dim)
        if torch.cuda.is_available():
            h0 = h0.cuda()
            c0 = c0.cuda()

        x = x.view(-1, seq_len, input_dim)
        z, _ = self.lstm(x, (h0, c0))
        y = self.lin(z[:, -1, :])
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
EPOCHS = 10
VIEW_INTERVAL = 200
for epoch in range(EPOCHS):

    running_loss = 0
    for i, data in enumerate(loader_train):
        x, y = data
        if torch.cuda.is_available():
            x, y = x.cuda(), y.cuda()
        optimizer.zero_grad()
        y_pred = model(x)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()

        # report loss
        running_loss += loss.item()
        if i % VIEW_INTERVAL == VIEW_INTERVAL - 1:
            print(f"Epoch {epoch+1}, iteration {i+1}; loss: {(running_loss / VIEW_INTERVAL):.3f}")
            running_loss = 0

            # testing
            with torch.no_grad():
                count_correct = 0
                count_total = 0
                for images, labels in loader_test:
                    if torch.cuda.is_available():
                        images, labels = images.cuda(), labels.cuda()
                    y_pred = model(Variable(images))
                    _, labels_pred = torch.max(y_pred.data, 1)
                    c = (labels_pred == labels).squeeze()
                    count_correct += c.sum().item()
                    count_total += len(c)
                print(f"  Test Acc: {100.0 * count_correct / count_total :.2f}%")
