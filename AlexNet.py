import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import nn
from torch.autograd import Variable
import pandas as pd
import matplotlib.pyplot as plt
from data_feed import *
pd.options.display.float_format = '{:.2f}'.format
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"



class AlexNet(nn.Module):
    def __init__(self ,init_weights=False,init_shape = None):
        super(AlexNet, self).__init__()


        '''
        
        '''

        self.features = nn.Sequential(
            nn.Conv1d(1 ,1, kernel_size=3, stride=1), # output channel: pic size;
            nn.ReLU(inplace=True),
            # nn.BatchNorm1d(96), # normalization
            #nn.BatchNorm1d(16),
            nn.MaxPool1d(kernel_size=3, stride=2),  # kernel 3x3; step 2x2 -> (30-3)/2 + 1 = 14x14

            nn.Conv1d(1,1, kernel_size=3, stride=1), # (14-3)/1 + 1 = 12 x 12
            nn.ReLU(inplace=True),
            # nn.BatchNorm1d(256),

            nn.MaxPool1d(kernel_size=3, stride=2),  # (12 - 3) /2 +1 = 5x5

            nn.Conv1d(1, 1, kernel_size=3, padding=1, stride=1),  # (5-3+2*1)/1+1=5x5

        )

        self.classifier = nn.Sequential(

            nn.Linear(6519, 3000),

            nn.Dropout(0.3),

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



def train(dataset,batch_size,epochs,device,**kwargs):

    trainLoader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    testLoader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)

    print("Done data contribution... move to GPU process")

    all_loss = []
    feature_size = kwargs["feature_size"]
    print("start modelling")
    model = AlexNet(init_shape=feature_size).to(device)

    criterion = nn.L1Loss()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001,weight_decay=0.00001)

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
    if kwargs["fig"] == True:
        figname = kwargs["figname"]
        plt.plot(range(len(all_loss)),all_loss)
        plt.xlabel("Round")
        plt.ylabel("Overall Loss")
        plt.title(figname)
        filename = kwargs["path"]+kwargs["figname"]+".jpg"
        plt.savefig(filename)

    return model


OUTPATH=""
def main():
    parser = argparse.ArgumentParser(description=INTRO)
    req_grp = parser.add_argument_group(title='Required')
    req_grp.add_argument('-1', '--train', type=str, help="Input train set.", required=True)
    req_grp.add_argument('-2', '--test', type=str, help="Input test set.", required=True)
    req_grp.add_argument('-t', '--trait', type=str, help="Input trait.", required=True)
    req_grp.add_argument('-o', '--output', type=str, help="Input output dir.", required=True)
    req_grp.add_argument('-s', '--sampleSize', type=str, help="input sample size", default="all")
    req_grp.add_argument('-r', '--region', type=bool, help="train by region?", default=False)

    args = parser.parse_args()
    if args.output[0] == "/":
        locat = '/' + args.output.strip('/') + '/'
    else:
        locat = args.output.strip('/') + '/'
    sub_folder = locat + "Train_{}_Test_{}_{}/".format(args.train,args.test,args.sampleSize)
    try:
        os.system("mkdir -p {}".format(locat))
        os.system("mkdir -p {}".format(sub_folder))
    except:
        print("folder exist.")


    global OUTPATH
    OUTPATH = sub_folder
    sample_size = args.sampleSize
    train_filepath = "E:/learning resource/PhD/sugarcane/" + args.train + "_" + args.trait + "_" + sample_size + ".csv"
    test_filepath = "E:/learning resource/PhD/sugarcane/" + args.test + "_" + args.trait + "_" + sample_size + ".csv"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))
    print("Loading data from directory")
    #trYs = torch.clone(torch.load("../complete_TrainData_traits.pt"))  # , dtype=torch.float32)
    train_data = pd.read_csv(train_filepath,sep="\t")
    print("Finish train data loading")
    test_data = pd.read_csv(test_filepath,sep="\t").sample(400)
    print("Finidsh test data loading")

    by_region = {}
    if "Region" in train_data.keys() and args.region is True:
        for region in pd.unique(train_data["Region"]):
            sub_train = train_data[train_data["Region"] == region]
            sub_test = test_data[test_data["Region"] == region]
            sub_test.drop(["Region"],inplace=True,axis=1)
            sub_train.drop(["Region"],inplace=True,axis=1)
            by_region[region] = (sub_train,sub_test)
    else:
        by_region["whole"] = (train_data,test_data)
        try:
            for data in by_region["whole"]:
                data.drop(["Region"],inplace=True)
        except:
            print("The data have no regions.")

    records = pd.DataFrame(columns=["train", "accuracy"])
    for region in by_region.keys():

        print("Now process {} data.".format(region))
        subset = by_region[region]
        feature_size = subset[0].shape[1]
        trYs = torch.tensor(np.array(subset[0][[args.trait]]).astype(float))
        print(trYs.shape)
        traits = [trYs[:, x].reshape(trYs.shape[0], 1) for x in range(1)]

        testYs = torch.tensor(np.array(subset[1][[args.trait]]).astype(float))
        test_traits = [testYs[:, x].reshape(testYs.shape[0], 1) for x in range(1)]

        # trYs = trYs.reshape(trYs.shape[0], 3).type(torch.float32)
        loss = nn.L1Loss()

        print(train_data.shape)
        trXgenos = torch.tensor(np.array(subset[0].drop([args.trait], axis=1)).astype(float))  # dtype=torch.float32)
        print(trXgenos.shape)
        trXgenos = trXgenos.reshape(trXgenos.shape[0], trXgenos.shape[1]).type(torch.float32)

        testXgenos = torch.tensor(np.array(subset[1].drop([args.trait], axis=1)).astype(float))
        testXgenos = testXgenos.reshape(testXgenos.shape[0], testXgenos.shape[1]).type(torch.float32)

        print("Finish loading")

        print("Start allocating dataset.")
        trainDatas = [Mydataset(trait, trXgenos) for trait in traits]

        testDatas = [Mydataset(trait, testXgenos) for trait in
                     test_traits]

        batch_size = 64
        epochs = 50
        train_models = []

        # names = ["CCSBlup","TCHBlup","FibreBlup"]
        for idx in range(len(traits)):
            BtestLoader = DataLoader(dataset=testDatas[idx], batch_size=batch_size, shuffle=False)
            # n = names[idx]
            train_name = "{}_{}".format(region,args.trait)
            m = train(trainDatas[idx], batch_size, epochs, device, fig=True, path=OUTPATH, figname=train_name,feature_size=feature_size)
            train_models.append(m)

            test_results = test(BtestLoader, m, loss, device)
            print(test_results)
            obv = test_results[0]
            pred = test_results[1]
            obv = obv.cpu().reshape(1, obv.shape[0])
            pred = pred.cpu().reshape(1, pred.shape[0])
            print(train_name)
            accuracy =  np.corrcoef(pred, obv)[0][1]
            print(accuracy)
            records = records.append({"train":train_name,"accuracy":accuracy},ignore_index=True)
        for i in range(len(train_models)):
            name = args.trait
            mm = train_models[i]
            print("saving")
            torch.save(mm, "{}{}_{}_model.pth".format(locat,region,name))

    print(records)
    records_name = OUTPATH + args.trait + "records.csv"
    records.to_csv(records_name, sep="\t")
    print("DONE")


if __name__ == "__main__":
    main()

#python AlexNet.py -1 2015 -2 2016 -t CCSBlup -o ../new_model/ -s 2000 -r True