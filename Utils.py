import torch
from torch import nn
from torch.utils import data as Data
import numpy as np


def shuffle(data):
     data = np.array(data)
     np.random.shuffle(data)
     return data

class Dataset_epoch(Data.Dataset):

    def __init__(self, datapath_data, fs=500):
        self.data = datapath_data
        self.N1 = []
        self.N2 = []
        self.N3 = []
        self.REM = []

        self.count_1 = 0
        self.count_2 = 0
        self.count_3 = 0
        self.count_R = 0

        for path in datapath_data:
            label = int(path.split("stage")[1][0])
            if label == 0:
                self.N1.append(path)
            elif label == 1:
                self.N2.append(path)
            elif label == 2:
                self.N3.append(path)
            else:
                self.REM.append(path)

        shuffle(self.N1)
        shuffle(self.N2)
        shuffle(self.N3)
        shuffle(self.REM)

        self.fs = fs


    def __len__(self):
        return len(self.data)


    def __getitem__(self, step):

        pick = step % 4
        if pick == 0:
            datapath = self.N1[self.count_1]
            self.count_1 += 1
            label = 0
            if self.count_1 == len(self.N1):
                self.count_1 = 0

        elif pick == 1:
            datapath = self.N2[self.count_2]
            self.count_2 += 1
            label = 1
            if self.count_2 == len(self.N2):
                self.count_2 = 0

        elif pick == 2:
            datapath = self.N3[self.count_3]
            self.count_3 += 1
            label = 2
            if self.count_3 == len(self.N3):
                self.count_3 = 0

        elif pick == 3:
            datapath = self.REM[self.count_R]
            self.count_R += 1
            label = 3
            if self.count_R == len(self.REM):
                self.count_R = 0

        data = torch.tensor(np.load(datapath), dtype=torch.float32)

        return data, label


class Dataset_epoch_FineTune(Data.Dataset):

    def __init__(self, datapath_data, fs=500):
        self.data = datapath_data
        self.fs = fs


    def __len__(self):
        return len(self.data)


    def __getitem__(self, step):
        datapath = self.data[step]
        label = int(datapath.split("stage")[1][0])

        if label == 0:
            label = 0
        elif label == 1:
            label = 1
        elif label == 2:
            label = 2
        else:
            label = 3

        data = torch.tensor(np.load(datapath), dtype=torch.float32)

        return data, label


class loss_func(nn.Module):

    def __init__(self, alpha, beta, device, weight=None):
        '''

        :param alpha: Weight of ce loss
        :param beta: Weight of mse loss
        :param device: GPU working device
        :param weight: Weight distribution of CE loss, in order to balance the
        '''
        super(loss_func, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.lossMSE = nn.MSELoss().to(device)
        self.device = device
        if not weight:
            self.lossCE = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array(weight)).float()).to(device)
        else:
            self.lossCE = nn.CrossEntropyLoss().to(device)


    def forward(self, x, y):
        lossce = self.lossCE(x, y)
        temp = []
        for i in x:
            t = 0
            for j in range(len(i)):
                t += (i[j] * j)
            temp.append(t)
        temp = torch.tensor(temp).to(self.device)
        lossmse = self.lossMSE(temp, y)
        loss = self.alpha * lossce + self.beta * lossmse
        return loss