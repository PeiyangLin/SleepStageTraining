import numpy as np
import pandas as pd
import glob
import time

DataPath1 = "../../Data/FL/*"
# DataPath2 = "../../Data/insight/*"
savePath = "../../Data/Slices_Norm_FL/"
fs = 500
duration = 30 * fs
sleep_dict = {"W": 0, "N1": 0, "N2": 1, "N3": 2, "R": 3}

subjects_FL = glob.glob(DataPath1)
FL_DataPath = []
FL_Label = []
index = []

for name in subjects_FL:
    indx = name[-3:]
    index.append(indx)
    FL_DataPath.append(name + "/%s_eeg_data.npy" % indx)
    FL_Label.append(name + "/hypno_30s.csv")

for name in range(len(FL_DataPath)):
    time0 = time.time()
    label_total = pd.read_csv(FL_Label[name])
    onsets = list(label_total["onset"])
    for i in range(len(onsets)):
        onsets[i] = int(onsets[i] * fs)
    description = list(label_total["description"])
    for i in range(len(description)):
        description[i] = sleep_dict[description[i]]

    data = np.load(FL_DataPath[name])
    shape = data.shape
    count = len(description)
    print(count)

    data_C3 = data[2]
    data_REOG = data[6]
    data_EMG = data[7]
    data = np.array([data_C3, data_REOG, data_EMG])

    for i in range(count):
        data_temp = data[:, onsets[i]:onsets[i] + duration]
        saveName = savePath + "%s-%04d-stage%d.npy" % (index[name], i + 1, description[i])
        np.save(saveName, data_temp)

    print("Done! Use Time: %.3f" % (time.time() - time0))
