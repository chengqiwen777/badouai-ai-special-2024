import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms


class Model:
    def __init__(self, net, cost, optimizer):
        self.net = net
        self.cost = self._create_cost(cost)
        self.optimizer = self._create_optimizer(optimizer)

    def _create_cost(self, cost):
        support_cost = {
            'CROSS_ENTROPY': nn.CrossEntropyLoss(),
            'MSE': nn.MSELoss()
        }

        return support_cost[cost]

    def _create_optimizer(self, optimizer, **rest):
        support_optimizer = {
            'SGD': optim.SGD(self.net.parameters(), lr=0.001, **rest),
            'ADAM': optim.Adam(self.net.parameters(), lr=0.001, **rest),
            'RMSP': optim.RMSprop(self.net.parameters(), lr=0.001, **rest)
        }

        return support_optimizer[optimizer]

    def train(self, train_loader, epochs=1):
        for e in range(epochs):
            running_loss = 0.0
            for i, data in enumerate(train_loader):
                inputs, labels = data  # inputs: [32, 1, 28, 28] labels:[32]
                print('inputs.shape:', inputs.shape)
                print('labels.shape:', labels.shape)
                # 它用于清除（或归零）所有被优化过的变量的梯度。在PyTorch的自动微分系统中，梯度是自动计算的，并且在默认情况下会累积，这意味着如果你在一个循环中多次调用 .backward()，梯度会被累加到已有的梯度上，而不是被替换。
                self.optimizer.zero_grad()

                outputs = self.net(inputs)
                loss = self.cost(outputs, labels)
                loss.backward()
                self.optimizer.step()
                print('loss:', loss)
                print('loss.item():', loss.item())
                running_loss += loss.item()
                print('i:', i)
                if i % 100 == 0:
                    print('[Epoch: %d, %.2f%%] loss: %.3f' %
                          (e + 1, (i + 1) * 1.0 / len(train_loader), running_loss / 100))
                    running_loss = 0.0

        print('Finished Training')

    def evaluate(self, test_loader):
        print('Evaluating')
        correct = 0
        total = 0
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                print('i:', i)
                images, labels = data

                outputs = self.net(images)
                print('outputs:', outputs)  # 还没输出
                pred = torch.argmax(outputs)
                print('pred', pred)

                predicted = torch.argmax(outputs, 1)
                print('predicted:', predicted)
                print('label:', labels)
                print('label.shape:', labels.shape)
                total += labels.size(0)
        #         correct += (predicted == labels).sum()
        # print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))


def mnist_load_data():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.,), (1,))])

    train_set = torchvision.datasets.MNIST(root='./data_mnist', train=True,
                                           download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=32,
                                               shuffle=True, num_workers=2)
    test_set = torchvision.datasets.MNIST(root='./data_mnist', train=False,
                                          download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=32,
                                              shuffle=True, num_workers=2)
    return train_loader, test_loader


class MnistNet(nn.Module):
    def __init__(self):
        super(MnistNet, self).__init__()
        self.fc1 = torch.nn.Linear(28 * 28, 512)
        self.fc2 = torch.nn.Linear(512, 512)
        self.fc3 = torch.nn.Linear(512, 10)

    def forward(self, x):
        # 改变张量的形状
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)
        return x


if __name__ == '__main__':
    net = MnistNet()
    model = Model(net, 'CROSS_ENTROPY', 'SGD')
    train_loader, test_loader = mnist_load_data()
    model.train(train_loader)
    model.evaluate(test_loader)
