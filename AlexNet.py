import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import nn
from torch.autograd import Variable
import pandas as pd
import matplotlib.pyplot as plt
pd.options.display.float_format = '{:.2f}'.format



class AlexNet(nn.Module):
    def __init__(self ,init_weights=False):
        super(AlexNet, self).__init__()


        '''

        data size:3channel x32x32
        1st convolutional layer, input channel=3, output channel=96, kernel size=3x3, step=1*1'''

        '''feature map size=(32-3)/1+1=30,feature map dimension 96*30*30'''

        self.features = nn.Sequential(
            nn.Conv1d(1 ,1, kernel_size=3, stride=1), # output channel: pic size;
            nn.ReLU(inplace=True),
            # nn.BatchNorm1d(96), # normalization
            nn.BatchNorm1d(1),
            nn.MaxPool1d(kernel_size=3, stride=2),  # kernel 3x3; step 2x2 -> (30-3)/2 + 1 = 14x14

            nn.Conv1d(1, 1, kernel_size=3, stride=1), # (14-3)/1 + 1 = 12 x 12
            nn.ReLU(inplace=True),
            # nn.BatchNorm1d(256),

            nn.MaxPool1d(kernel_size=3, stride=2),  # (12 - 3) /2 +1 = 5x5

            nn.Conv1d(1, 1, kernel_size=3, padding=1, stride=1),  # (5-3+2*1)/1+1=5x5

        )

        self.classifier = nn.Sequential(

            nn.Linear(6519, 3000),

            nn.Dropout(0.5),

            nn.Linear(3000, 1),

        )
        if init_weights:
            self._initialize_weights()


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):

        x = x.unsqueeze(1)

        out = self.features(x)

        out = torch.flatten(out ,start_dim=1)

        # out = out.reshape(-1, 256 * 2 * 2)

        out = self.classifier(out)

        return out


def test(dataloader, model, loss_fn, device):

    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    y,pred = 0,0
    test_loss, correct = 0, 0
    with torch.no_grad():
        for i, (images, labels) in enumerate(dataloader): # enumerate

            images =Variable(images)
            labels =Variable(labels)
            # print(labels)
            # Forward pass
            # outputs = model(images)
            # loss = criterion(outputs, labels)

            X, y = images.to(device), labels.to(device)
            pred = model(X)

            test_loss += loss_fn(pred, y)

            #correct += (pred == y).sum().item()
            # correct += (pred == y).type(torch.max).sum().item()
            size += len(labels)
        correct /= size
        test_loss /= num_batches

    print(f"Test Error: \n  Avg loss: {test_loss:>8f} \n")
    return y, pred


class Mydataset(Dataset):

    def __init__(self, trYs, trXgenos, transform=None):
        self.traits = trYs
        self.trXgenos = trXgenos

        assert len(self.traits) == len(self.trXgenos), "samples not match"

        self.transform = transform

        self.sample_set = []
        for idx in range(len(self.trXgenos)):
            self.sample_set.append((self.trXgenos[idx], self.traits[idx]))

    def __getitem__(self, index):
        genes, traits = self.sample_set[index]
        if self.transform is not None:
            genes = self.transform(genes)

        return genes, traits

    def __len__(self):
        return len(self.trXgenos)



def train(dataset,batch_size,epochs,device,name):

    trainLoader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    testLoader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)

    print("Done data contribution... move to GPU process")

    all_loss = []

    print("start modelling")
    model = AlexNet().to(device)

    criterion = nn.L1Loss()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    total_step = len(trainLoader)
    epoch_i = 0
    for epoch in range(epochs):
        model.train()
        ii = 0
        temp_ls = 0
        for i, (images, labels) in enumerate(trainLoader):
            optimizer.zero_grad()
            images = Variable(images).to(device)
            labels = Variable(labels).to(device)
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 10 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, epochs, i + 1, total_step, loss.item()))
                temp_ls = loss.item()
        all_loss.append(temp_ls)
        model.eval()
        epoch_i += ii
        print("test")
        test(testLoader, model, criterion, device)
    plt.plot(range(len(all_loss)),all_loss)
    plt.xlabel("Round")
    plt.ylabel("Overall Loss")
    plt.title(name)
    plt.show()
    return model


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))
    print("Loading data from directory")
    trYs = torch.clone(torch.load("../complete_TrainData_traits.pt"))  # , dtype=torch.float32)
    traits = [trYs[:, x].reshape(trYs.shape[0], 1) for x in range(3)]
    # trYs = trYs.reshape(trYs.shape[0], 3).type(torch.float32)
    loss = nn.L1Loss()
    trXgenos = torch.clone(torch.load("../complete_TrainData_genos.pt"))  # , dtype=torch.float32)
    trXgenos = trXgenos.reshape(trXgenos.shape[0], trXgenos.shape[1]).type(torch.float32)
    print(trYs.shape)
    print(trXgenos.shape)

    print("Finish loading")

    print("Start allocating dataset.")
    trainDatas = [Mydataset(trait, trXgenos) for trait in traits]

    # testDatas = trainDatas # [Mydataset(trait, trXgenos) for trait in traits]

    batch_size = 64
    epochs = 50
    train_models = []
    names = ["CCSBlup","TCHBlup","FibreBlup"]
    for idx in range(len(traits)):
        BtestLoader = DataLoader(dataset=trainDatas[idx], batch_size=batch_size, shuffle=False)
        n = names[idx]
        m = train(trainDatas[idx],batch_size,epochs,device,n)
        train_models.append(m)

        test_results = test(BtestLoader,m,loss,device)
        print(test_results)
    for i in range(len(train_models)):
        name = ["CCSBlup","TCHBlup","FibreBlup"][i]
        mm = train_models[i]
        torch.save(mm,"../models/{}_model.pth".format(name))
    print("DONE")


if __name__ == "__main__":
    main()