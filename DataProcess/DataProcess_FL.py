import numpy as np
import pandas as pd
import glob
import time
import math


def channel_pick(channelPath):
    channel = np.load(channelPath)
    print(channel)
    EEG_Pick = 'C3-M2'
    EOG_Pick = 'REOG'
    EMG_Pick = 'LEMG'

    EEG_Pick = int(np.where(channel == EEG_Pick)[0])
    EOG_Pick = int(np.where(channel == EOG_Pick)[0])
    EMG_Pick = int(np.where(channel == EMG_Pick)[0])

    print("Pick Channel:", channel[EEG_Pick], channel[EOG_Pick], channel[EMG_Pick])
    return EEG_Pick, EOG_Pick, EMG_Pick


def DC_remove(data):
    data = data - np.mean(data)
    return data


def stage_Cal(label):
    count_W_N1 = label.count(0)
    count_N2 = label.count(1)
    count_N3 = label.count(2)
    count_REM = label.count(3)

    stage_distribute = np.array([count_W_N1, count_N2, count_N3, count_REM])
    stage_max = max(stage_distribute)
    stage_index = np.where(stage_distribute == stage_max)[0]

    result = 0

    if 0 in stage_index:
        result = 0
    elif 3 in stage_index:
        result = 3
    elif 2 in stage_index:
        result = 2
    elif 1 in stage_index:
        result = 1

    return result


DataPath_All = "../../Data/FL/*"
savePath = "../../Data/Slice_FL/"
fs = 500
duration = 30
stride = 3
sleep_dict = {"W": 0, "N1": 0, "N2": 1, "N3": 2, "R": 3}

duration *= fs
stride *= fs

subjects = glob.glob(DataPath_All)[3:]
DataPath = []
Label = []
Channel = []
index = []

for name in subjects:
    indx = name[-3:]
    index.append(indx)
    DataPath.append(name + "/%s_eeg_data.npy" % indx)
    Label.append(name + "/hypno_30s.csv")
    Channel.append(name + "/%s_eeg_channel.npy" % indx)

# Training Set
for name in range(len(DataPath) - 3):
    time0 = time.time()
    data = np.load(DataPath[name])
    label_total = pd.read_csv(Label[name])
    onsets = list(label_total["onset"])
    description = list(label_total["description"])

    label = []

    for i in range(len(description)):
        stage_temp = sleep_dict[description[i]]
        for _ in range(duration):
            label.append(stage_temp)
    total_len = len(label)

    EEG_Pick, EOG_Pick, EMG_Pick = channel_pick(Channel[name])

    data_C3 = data[EEG_Pick][:total_len]
    data_REOG = data[EOG_Pick][:total_len]
    data_EMG = data[EMG_Pick][:total_len]

    data_C3 = DC_remove(data_C3)
    data_REOG = DC_remove(data_REOG)
    data_EMG = DC_remove(data_EMG)

    data = np.array([data_C3, data_REOG, data_EMG])

    num_windows = math.floor((total_len - duration) / stride) + 1
    print("Number of window:", num_windows)

    start = 0
    i = 0

    while start + duration <= total_len:
        end = start + duration
        data_temp = data[:, start:end]
        label_window = label[start:end]
        label_window = stage_Cal(label_window)
        saveName = savePath + "Train/%s-%05d-stage%d.npy" % (index[name], i + 1, label_window)
        # print(saveName)
        np.save(saveName, data_temp)
        start += stride
        i += 1

    print("Done! Use Time: %.3f" % (time.time() - time0))

# Testing Set
subjects = glob.glob(DataPath_All)[:3]
DataPath = []
Label = []
Channel = []
index = []

for name in subjects:
    indx = name[-3:]
    index.append(indx)
    DataPath.append(name + "/%s_eeg_data.npy" % indx)
    Label.append(name + "/hypno_30s.csv")
    Channel.append(name + "/%s_eeg_channel.npy" % indx)


for name in range(3):
    time0 = time.time()
    data = np.load(DataPath[name])
    label_total = pd.read_csv(Label[name])
    onsets = list(label_total["onset"])
    description = list(label_total["description"])

    label = []

    for i in range(len(description)):
        stage_temp = sleep_dict[description[i]]
        for _ in range(duration):
            label.append(stage_temp)
    total_len = len(label)

    EEG_Pick, EOG_Pick, EMG_Pick = channel_pick(Channel[name])

    data_C3 = data[EEG_Pick][:total_len]
    data_REOG = data[EOG_Pick][:total_len]
    data_EMG = data[EMG_Pick][:total_len]

    data_C3 = DC_remove(data_C3)
    data_REOG = DC_remove(data_REOG)
    data_EMG = DC_remove(data_EMG)

    data = np.array([data_C3, data_REOG, data_EMG])

    num_windows = math.floor((total_len - duration) / stride) + 1
    print("Number of window:", num_windows)

    start = 0
    i = 0

    while start + duration <= total_len:
        end = start + duration
        data_temp = data[:, start:end]
        label_window = label[start:end]
        label_window = stage_Cal(label_window)
        saveName = savePath + "Test/%s-%05d-stage%d.npy" % (index[name], i + 1, label_window)
        np.save(saveName, data_temp)
        start += stride
        i += 1

    print("Done! Use Time: %.3f" % (time.time() - time0))
