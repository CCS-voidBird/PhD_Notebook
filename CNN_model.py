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

train_loader=torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)

test_loader=torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False)




# Create data loaders.
#train_dataloader = DataLoader(training_data, batch_size=batch_size)
#test_dataloader = DataLoader(test_data, batch_size=batch_size)


class AlexNet(nn.Module):
    def __init__(self,init_weights=False):
        super(AlexNet, self).__init__()


        '''
        
        data size:3channel x32x32
        1st convolutional layer, input channel=3, output channel=96, kernel size=3x3, step=1*1'''

        '''feature map size=(32-3)/1+1=30,feature map dimension 96*30*30'''

        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=(3,3), stride=(1,1)), # output channel: pic size;
            nn.ReLU(inplace=True),
            #nn.BatchNorm2d(96), # normalization

            nn.MaxPool2d(kernel_size=3, stride=2),  # kernel 3x3; step 2x2 -> (30-3)/2 + 1 = 14x14

            nn.Conv2d(96, 256, kernel_size=(3,3), stride=(1,1)), # (14-3)/1 + 1 = 12 x 12
            nn.ReLU(inplace=True),
            #nn.BatchNorm2d(256),

            nn.MaxPool2d(kernel_size=3, stride=2),  # (12 - 3) /2 +1 = 5x5

            nn.Conv2d(256, 384, kernel_size=3, padding=1, stride=1),  # (5-3+2*1)/1+1=5x5
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # (5-3)/2+1=2
        )

        self.classifier = nn.Sequential(

            nn.Linear(1024, 2048),

            nn.Dropout(0.5),

            nn.Linear(2048, 2048),

            nn.Dropout(0.5),

            nn.Linear(2048, 10)
        )
        if init_weights:
            self._initialize_weights()


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')  # 何教授方法
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)  # 正态分布赋值
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.features(x)

        out = torch.flatten(out,start_dim=1)

        #out = out.reshape(-1, 256 * 2 * 2)

        out = self.classifier(out)

        """
        out = F.relu(self.linear1(out))

        out = self.dropout1(out)

        out = F.relu(self.linear2(out))

        out = self.dropout2(out)

        out = self.linear3(out)
        """
        return out

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)

    test_loss, correct = 0, 0
    with torch.no_grad():
        for i, (images, labels) in enumerate(dataloader): # enumerate

            images=Variable(images)
            labels=Variable(labels)
            #print(labels)
            # Forward pass
            #outputs = model(images)
            #loss = criterion(outputs, labels)

            X, y = images.to(device), labels.to(device)
            pred = model(X)
            predict = torch.max(pred,dim=1)[1]
            #print("##################")
            #print(predict)
            test_loss += loss_fn(pred, y).item()
            correct += (predict == y).sum().item()
            #correct += (pred == y).type(torch.max).sum().item()
            size += len(labels)
        correct /= size
        test_loss /= num_batches

    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

model = AlexNet().to(device)

#summary.summary(model, input_size=(3, 32, 32), batch_size=128, device="cpu")

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)

total_step = len(train_loader)
epochs = 30
for epoch in range(epochs):
    model.train()
    for i, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        images=Variable(images).to(device)
        labels=Variable(labels).to(device)
        # Forward pass
        outputs = model(images)

        loss = criterion(outputs, labels)

        # Backward and optimize
        #optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, epochs, i + 1, total_step, loss.item()))
    #print("temp train results: ")
    #print(torch.max(outputs, dim=1)[1])
    #print(labels)
    model.eval()
    print("test")
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

