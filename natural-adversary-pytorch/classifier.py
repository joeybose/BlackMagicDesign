import torch
import torch.nn as nn
import torch.nn.functional as F
from dataloader import get_loader
import time
import os

# Classifier.
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(256, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

if __name__ == '__main__':

    # Create directory if it doesn't exist.
    if not os.path.exists('./classifier/lenet'):
        os.makedirs('./classifier/lenet')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, test_loader = get_loader('mnist', 32)

    net = LeNet().to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)

    # Start training.
    print('Start training the classifier...')
    start_time = time.time()
    for i in range(1000):
        for j, (images, labels) in enumerate(train_loader):

            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)

            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Print out training information.
        elapsed_time = time.time() - start_time
        print('Elapsed time [{:.4f}], Iteration [{}/{}], Loss: {:.4f}'.format(
               elapsed_time, i+1, 1000, loss.item()
        ))

        # Save model checkpoints.
        if (i+1) % 100 == 0:
            model_path = './classifier/lenet/{}_lenet.ckpt'.format(i+1)
            torch.save(net.state_dict(), model_path)
            print('Saved model checkpoints into {}...'.format(model_path))
