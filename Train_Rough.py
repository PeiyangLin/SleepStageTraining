import torch

import  Utils
from Utils import loss_func

from torch.utils import data as Data
import numpy as np
import pandas as pd
import os
import time
import glob


class Para_Train:

    def __init__(self):
        # Basic Para
        self.Epochs = 11
        self.BatchSize = 256
        self.device = 0
        self.checkpoint = 5

        self.Model = "MTHead"
        self.fs = 256

        # Oasis
        indx = "2023_08_15_01"
        self.database = "../Data/Slices_Norm_FL/"
        self.savepath = "Model/%s/modelSave/%s" % (self.Model, indx)
        self.grade = 4

        # Model Path
        self.modelpath = "Model.%s.Model" % self.Model

        # Model Load
        self.isLoad = False
        load_indx = "2023_03_08_01_4Stage"
        self.loadPath = "Model/%s/modelSave/%s" % (self.Model, load_indx)

        # Learning Para
        self.LearningRate = 6e-4
        self.alpha = 0.8
        self.beta = 1 - self.alpha
        self.weight = [1, 1, 1, 1]

    def readme(self):
        RM = {"Epochs": self.Epochs, "BatchSize": self.BatchSize,
              "Device": self.device, "Model": self.Model,
              "savepath": self.savepath, "isLoad": self.isLoad,
              "LearningRate": self.LearningRate, "Alpha": self.alpha,
              "Beta": self.beta, "Weight": self.weight}

        data = pd.DataFrame.from_dict(RM, orient="index")

        data.to_csv(self.savepath + "/Parameter.csv")

        for key in RM:
            print("%s: %s" % (key, RM[key]))


def Accurate(label, pred):
    predict = pred.argmax(dim=1)
    length = torch.tensor(len(predict)).to(device)
    train_correct = (predict == label).sum()
    accuracy = train_correct / length
    return accuracy, length


def datacount(database):
    label = []
    label_0 = label_1 = label_2 = label_3 = label_4 = 0

    for path in database:
        temp = int(path.split("stage")[1][0])
        label.append(temp)
        if temp == 0:
            label_0 += 1
        elif temp == 1:
            label_1 += 1
        elif temp == 2:
            label_2 += 1
        elif temp == 3:
            label_3 += 1
        elif temp == 4:
            label_4 += 1
        else:
            pass
    label = np.array(label)
    length = len(label)
    print("N0: %d %.3f\nN1: %d %.3f\nN2: %d %.3f\nN3: %d %.3f\nREM: %d %.3f\nTotal: %d"
          % (label_0, label_0 / length, label_1, label_1 / length, label_2, label_2 / length, label_3, label_3 / length,
             label_4, label_4 / length,
             length))


def train_epoch(datas):
    model.train()
    with torch.enable_grad():
        loss_sum = []
        acc_epoch = []
        start = time.perf_counter()
        for step, (x, y) in enumerate(datas):
            x = x.to(device)
            y = y.to(device)

            y_pred = model(x)

            acc, _ = Accurate(y, y_pred)
            acc_epoch.append([step, acc.item()])

            loss = loss_f(y_pred, y)
            loss_sum.append([step, loss.item()])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            progress = (step / len(datas)) * 100
            finsh = "▓" * int(progress / 5)
            need_do = "-" * (20 - int(progress / 5))
            dur = time.perf_counter() - start
            print("{:^3.0f}%[{}->{}]{:.2f}s Loss:{:.5f} Acc:{:.2f}".format(progress, finsh, need_do, dur, loss.item(),
                                                                           acc))

        #             # debug
        #             if step==10:
        #                 break

        return loss_sum, acc_epoch, dur


def recall(ind, pred_label, y_label):
    TP = 0
    TN = 0
    FP = 0
    FN = 0

    total = 0
    for i in range(len(pred_label)):
        if pred_label[i] == ind:
            total += 1
        if (pred_label[i] == y_label[i]) and (pred_label[i] == ind):
            TP += 1
        elif (pred_label[i] == y_label[i]) and (pred_label[i] != ind):
            TN += 1
        elif (pred_label[i] != y_label[i]) and (pred_label[i] == ind):
            FP += 1
        elif (pred_label[i] != y_label[i]) and (pred_label[i] != ind):
            FN += 1

    try:
        TPR = TP / (TP + FP)
    except ZeroDivisionError:
        TPR = 0

    try:
        FPR = FP / (FP + TN)
    except ZeroDivisionError:
        FPR = 0

    try:
        RC = TP / (TP + FN)
    except ZeroDivisionError:
        RC = 0

    try:
        F = (TPR * RC * 2) / (TPR + RC)
    except ZeroDivisionError:
        F = 0

    print(("TPR: %.3f\nFPR: %.3f\nRC: %.3f\nF: %.3f") % (TPR, FPR, RC, F))


def test_epoch(datas):
    print("Evaluation:")
    model.eval()
    with torch.no_grad():
        loss_sum = []
        acc_epoch = []

        pred_label = []
        y_label = []

        start = time.perf_counter()
        for step, (x, y) in enumerate(datas):
            x = x.to(device)
            y_label.append(y.item())
            y = y.to(device)

            y_pred = model(x)
            pred_label.append(torch.argmax(y_pred, dim=1).item())

            acc, _ = Accurate(y, y_pred)
            acc_epoch.append([step, acc.item()])

            loss = loss_f(y_pred, y)
            loss_sum.append([step, loss.item()])

            progress = (step / len(datas)) * 100
            finsh = "▓" * int(progress / 5)
            need_do = "-" * (20 - int(progress / 5))
            dur = time.perf_counter() - start
            print("{:^3.0f}%[{}->{}]{:.2f}s Loss:{:.5f} Acc:{:.2f}".format(progress, finsh, need_do, dur, loss.item(), acc))

        print()
        for i in range(P.grade):
            print("ind: %d" % i)
            recall(i, pred_label, y_label)
            print()
    return loss_sum, acc_epoch, dur

#     return label

torch.backends.cudnn.benchmark = True
P = Para_Train()

# Basic para set
isLoad = P.isLoad
device = P.device

Epochs = P.Epochs
BatchSize = P.BatchSize
checkpoint = P.checkpoint

# Oasis and save path
fs = P.fs
now = time.time()

dataPath_train = P.database + "Train/*"
dataPath_train = np.array(sorted(glob.glob(dataPath_train)))

dataPath_test = P.database + "Test/*"
dataPath_test = np.array(sorted(glob.glob(dataPath_test)))

# Checking and Making Dictionary
savePath = P.savepath
isExists = os.path.exists(savePath)

if not isExists:
    os.mkdir(savePath)
    os.mkdir(savePath + "/model")
    os.mkdir(savePath + "/loss")
    os.mkdir(savePath + "/checkFrame")
    os.mknod(savePath + "/readme.txt")
readme = open(savePath + "/readme.txt", "w")
P.readme()
print("Oasis:")
print("Training Set: ", dataPath_train.shape[0])
print(datacount(dataPath_train))
print()
print("Testing Set: ", dataPath_test.shape[0])
print(datacount(dataPath_test))

# DataSet
train_sets_unload = Utils.Dataset_epoch(dataPath_train)
train_sets = Data.DataLoader(dataset=train_sets_unload, batch_size=BatchSize, shuffle=True)

test_sets_unload = Utils.Dataset_epoch_FineTune(dataPath_test)
test_sets = Data.DataLoader(dataset=test_sets_unload, batch_size=1, shuffle=False)

# Model Set
torch.cuda.set_device(device)
exec("from %s import Model_total" % P.modelpath)
model = Model_total().to(device)

if isLoad:
    load_path = P.loadPath + "/model/*.pth"
    load_path = sorted(glob.glob(load_path))[-1]
    print("Loading model", load_path)
    model.load_state_dict(torch.load(load_path))
    print("Done")

optimizer = torch.optim.Adam(model.parameters(), lr=P.LearningRate, weight_decay=1e-5)
loss_f = loss_func(P.alpha, P.beta, device=device, weight=P.weight)

# Training
loss_train = []
loss_test = []
loss_rerun = []

acc_train = []
acc_test = []
acc_rerun = []


for epoch in range(Epochs):
    print("Batch %03d/%03d" % (epoch + 1, Epochs))
    loss, acc, dur = train_epoch(train_sets)
    loss = np.array(loss)
    acc = np.array(acc)

    modelname = savePath + '/model/check%04d.pth' % epoch
    torch.save(model.state_dict(), modelname)

    temp_acc = []
    temp_loss = []
    for i in range(len(acc)):
        temp_acc.append(acc[i, 1])
        temp_loss.append(loss[i, 1])
    print("\nBatch %03d done, use time %.2f, avg loss: %.3f, avg acc: %.3f, max acc: %.3f" % (
    epoch + 1, dur, np.mean(temp_loss),
    np.mean(temp_acc) * 100, np.max(temp_acc) * 100))
    if ((epoch + 1) % checkpoint) == 0:
        print("Epoch %03d, testing:" % (epoch + 1))
        loss_test, acc_test, dur_test = test_epoch(test_sets)
        temp_acc_test = []
        temp_loss_test = []
        acc_test = np.array(acc_test)
        loss_test = np.array(loss_test)
        for i in range(len(acc_test)):
            temp_acc_test.append(acc_test[i, 1])
            temp_loss_test.append(loss_test[i, 1])
        np.save(savePath + "/loss/loss_test%03d.npy" % (epoch + 1), loss_test)
        np.save(savePath + "/loss/acc_test%03d.npy" % (epoch + 1), acc_test)
        print("\nTest Batch %03d done, use time %.2f, avg loss: %.3f, avg acc: %.3f" % (
        epoch + 1, dur_test, np.mean(temp_loss_test),
        np.mean(temp_acc_test) * 100))

    np.save(savePath + "/loss/loss%03d.npy" % (epoch + 1), loss)
    np.save(savePath + "/loss/acc%03d.npy" % (epoch + 1), acc)