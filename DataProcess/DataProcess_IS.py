import numpy as np
import pandas as pd
import glob
import time

DataPath2 = "../../Data/insight/*night*"
savePath2 = "../../Data/Slices_IS/"
fs = 500
duration = 30 * fs
sleep_dict = {"W": 0, "N1": 0, "N2": 1, "N3": 2, "R": 3}

subjects_IS = glob.glob(DataPath2)[:-3]
IS_DataPath = []
IS_Label = []
index = []

for name in subjects_IS:
    indx = name.split("sub")[1][:2]
    index.append(indx)
    IS_DataPath.append(name + "/sub%s_night_eeg_data.npy" % indx)
    IS_Label.append(name + "/hypno_30s.csv")

for name in range(len(IS_DataPath)):
    time0 = time.time()
    label_total = pd.read_csv(IS_Label[name])
    onsets = list(label_total["onset"])
    for i in range(len(onsets)):
        onsets[i] = int(onsets[i] * fs)
    description = list(label_total["description"])
    for i in range(len(description)):
        description[i] = sleep_dict[description[i]]

    data = np.load(IS_DataPath[name])
    shape = data.shape
    count = len(description)
    print(count)

    data_C3 = data[2]
    data_REOG = data[6]
    data_EMG = data[7]
    data = np.array([data_C3, data_REOG, data_EMG])

    for i in range(count):
        data_temp = data[:, onsets[i]:onsets[i] + duration]
        saveName = savePath2 + "Train/IS0%s-%04d-stage%d.npy" % (index[name], i + 1, description[i])
        np.save(saveName, data_temp)

    print("Done! Use Time: %.3f" % (time.time() - time0))

subjects_IS = glob.glob(DataPath2)[-3:]
IS_DataPath = []
IS_Label = []
index = []

for name in subjects_IS:
    indx = name.split("sub")[1][:2]
    index.append(indx)
    IS_DataPath.append(name + "/sub%s_night_eeg_data.npy" % indx)
    IS_Label.append(name + "/hypno_30s.csv")

for name in range(len(IS_DataPath)):
    time0 = time.time()
    label_total = pd.read_csv(IS_Label[name])
    onsets = list(label_total["onset"])
    for i in range(len(onsets)):
        onsets[i] = int(onsets[i] * fs)
    description = list(label_total["description"])
    for i in range(len(description)):
        description[i] = sleep_dict[description[i]]

    data = np.load(IS_DataPath[name])
    shape = data.shape
    count = len(description)
    print(count)

    data_C3 = data[2]
    data_REOG = data[6]
    data_EMG = data[7]
    data = np.array([data_C3, data_REOG, data_EMG])

    for i in range(count):
        data_temp = data[:, onsets[i]:onsets[i] + duration]
        saveName = savePath2 + "Test/IS0%s-%04d-stage%d.npy" % (index[name], i + 1, description[i])
        np.save(saveName, data_temp)

    print("Done! Use Time: %.3f" % (time.time() - time0))
