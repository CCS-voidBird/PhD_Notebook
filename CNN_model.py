import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt

#example case https://zhuanlan.zhihu.com/p/250120949

batch_size = 64
train_dataset=torchvision.datasets.CIFAR10(root='../DataSet/',train=True,download=True,transform=transforms.ToTensor())

test_dataset=torchvision.datasets.CIFAR10(root='../DataSet/',train=False,transform=transforms.ToTensor())

'''数据装载'''
train_loader=torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)

test_loader=torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False)




# Create data loaders.
#train_dataloader = DataLoader(training_data, batch_size=batch_size)
#test_dataloader = DataLoader(test_data, batch_size=batch_size)


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()


        '''
        
        data size:3channel x32x32
        1st convolutional layer, input channel=3, output channel=96, kernel size=3x3, step=1*1'''

        '''feature map size=(32-3)/1+1=30,feature map dimension 96*30*30'''

        self.conv1 = nn.Conv2d(3, 96, kernel_size=(3,3), stride=(1,1)) # output channel: pic size;

        self.bn1 = nn.BatchNorm2d(96) # normalization

        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)  # kernel 3x3; step 2x2 -> (30-3)/2 + 1 = 14x14

        self.conv2 = nn.Conv2d(96, 256, kernel_size=(3,3), stride=(1,1)) # (14-3)/1 + 1 = 12 x 12

        self.bn2 = nn.BatchNorm2d(256)

        '''这样经过第二层池化层之后，得到的feature map的大小为(12-3)/2+1=5,所以feature map的维度为256*5*5'''

        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)  # (12 - 3) /2 +1 = 5x5

        '''第三层卷积层，卷积核为3*3，通道数为384，步距为1，前一层的feature map的大小为5*5，通道数为256个'''

        '''这样经过第一层卷积层之后，得到的feature map的大小为(5-3+2*1)/1+1=5,所以feature map的维度为384*5*5'''

        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, padding=1, stride=1)  # (5-3+2*1)/1+1=5x5

        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, padding=1, stride=1)

        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, padding=1, stride=1)

        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2)  # (5-3)/2+1=2

        self.linear1 = nn.Linear(1024, 2048)

        self.dropout1 = nn.Dropout(0.5)

        self.linear2 = nn.Linear(2048, 2048)

        self.dropout2 = nn.Dropout(0.5)

        self.linear3 = nn.Linear(2048, 10)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.pool1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)
        out = self.pool2(out)

        out = F.relu(self.conv3(out))

        out = F.relu(self.conv4(out))

        out = F.relu(self.conv5(out))

        out = self.pool3(out)

        out = out.reshape(-1, 256 * 2 * 2)

        out = F.relu(self.linear1(out))

        out = self.dropout1(out)

        out = F.relu(self.linear2(out))

        out = self.dropout2(out)

        out = self.linear3(out)

        return out

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for i, (images, labels) in enumerate(dataloader): # enumerate

            images=Variable(images)
            labels=Variable(labels)
            # Forward pass
            #outputs = model(images)
            #loss = criterion(outputs, labels)

            X, y = images.to(device), labels.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            size += len(labels)
            print(correct)
            print(size)
        correct /= size
        test_loss /= num_batches

    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

model = AlexNet().to(device)

#summary.summary(model, input_size=(3, 32, 32), batch_size=128, device="cpu")

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

total_step = len(train_loader)

for epoch in range(10):
    for i, (images, labels) in enumerate(train_loader):

        images=Variable(images)
        labels=Variable(labels)
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, 10, i + 1, total_step, loss.item()))

    test(test_loader,model,criterion)


# Get cpu or gpu device for training.


# Define model
"""
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.conv1 = nn.Conv2d(1,6,5)
        self.conv2 = nn.Conv2d(6,16,5)
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.flatten(x)
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # if the size is square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        logits = self.linear_relu_stack(x)
        return logits
"""

