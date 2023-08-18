import numpy as np
import pandas as pd
import os
import mne
import yasa
import time
import SleepStageDetect
from tqdm import tqdm
import warnings
import glob
import torch

warnings.filterwarnings("ignore")

def timecal(useTime):
    m, s = divmod(useTime, 60)
    h, m = divmod(m, 60)
    return "%02dh %02dm %.3fs" % (h, m, s)


def All_Processing(dataPath, labelPath, indx, warmingTime, Time_Sld, C3, REog, EMG, dataBase):
    print("Processing %s Data, index %03d\n" % (dataBase, indx))
    now_init = time.time()

    # MASS channel init

    fs = 500
    model_stride = 30
    yasa_stride = 30
    realtime_stride = 3
    Time_Sld = Time_Sld * fs
    sleep_dict = {"W": 0, "N1": 0, "N2": 2, "N3": 3, "R": 4}
    # Sleep Stage mapping
    mapping_yasa = {"W": 0, "N1": 0, "N2": 2, "N3": 3, "R": 4}
    mapping_model = {"W/N1": 0, "N2": 2, "N3": 3, "REM": 4}

    # Label
    print("Data Loading...")
    now = time.time()
    label_total = pd.read_csv(labelPath)
    onsets = list(label_total["onset"])
    for i in range(len(onsets)):
        onsets[i] = int(onsets[i] * fs)
    label = list(label_total["description"])

    for i in range(len(label)):
        label[i] = sleep_dict[label[i]]

    label_long = []
    for l in label:
        for _ in range(model_stride):
            label_long.append(l)

    count = len(label)
    label = label_long

    # Data Load
    data = np.load(dataPath)
    total_length = []
    total_length.append(len(label))

    # Paths init
    SaveLabel = "BenchMark/Result/%s/%s_%03d" % (dataBase, dataBase, indx)
    if not os.path.exists(SaveLabel):
        os.mkdir(SaveLabel)
    print("Done! Use time: %s\n" % timecal(time.time() - now))

    # Predict offline yasa
    print("Predicting offline data by yasa...")
    now = time.time()
    data_C3 = data[C3]
    data_REOG = data[REog]
    data_EMG = data[EMG]
    data_yasa = np.array([data_C3, data_REOG, data_EMG])
    info = mne.create_info(ch_names=['C3', 'REOG', "EMG"],
                           ch_types='eeg',
                           sfreq=fs)
    raw_yasa = mne.io.RawArray(data_yasa, info, verbose=False)
    sls = yasa.SleepStaging(raw_yasa, eeg_name='C3', eog_name='REOG', emg_name="EMG")
    pred_yasa = sls.predict()
    print(pred_yasa)
    offline_yasa = []
    for p in pred_yasa:
        temp_stage = mapping_yasa[p]
        for _ in range(yasa_stride):
            offline_yasa.append(temp_stage)

    offline_yasa = np.array(offline_yasa)
    total_length.append(len(offline_yasa))
    print("Done! Use time: %s\n" % timecal(time.time() - now))

    # Predict Offline Model
    print("Predicting offline data by our model...")
    now = time.time()
    # Data Splitting and Model Init
    data_slices = []
    data_my = np.array([data_C3, data_REOG, data_EMG])

    for i in tqdm(range(count), desc="Data Splitting"):
        data_temp = data_my[:, onsets[i]:onsets[i] + Time_Sld]
        data_slices.append(data_temp)

    data_slices = torch.tensor(data_slices)
    net = SleepStageDetect.SleepStageModel()

    # Offline Predicting
    offline_model = []
    for i in tqdm(range(len(data_slices)), desc="Offline Predicting"):
        dat = data_slices[i]
        pred_model, _ = net.predict(dat)
        for _ in range(model_stride):
            offline_model.append(mapping_model[pred_model])
    offline_model = np.array(offline_model)
    total_length.append(len(offline_model))
    print(len(offline_model))
    print("Done! Use time: %s\n" % timecal(time.time() - now))

    data_slices = None
    data_my = None
    data_yasa = None

    # Offline Data Saving
    length_min = min(total_length)
    print(length_min)
    offline_yasa = offline_yasa[:length_min]
    offline_model = offline_model[:length_min]

    np.save(SaveLabel + "/GroundTruth.npy", label)
    np.save(SaveLabel + "/Offline_yasa.npy", offline_yasa)
    np.save(SaveLabel + "/Offline_model.npy", offline_model)
    print("Done! Use time: %s\n" % timecal(time.time() - now))

    useTime_all = timecal(time.time() - now_init)
    print("Realtime data, index %03d done! Use time: %s\n" % (indx, useTime_all))

# Only Hyperparameters you will change:
warmingTime = 3600
Time_Sld = 30
C3 = 2
REog = 6
EMG = 7

# indx = 20
# DataPath = "I:/DataSets/CIBR_data/FL/*"
DataPath = "I:/DataSets/CIBR_data/insight/*"

DataPath = glob.glob(DataPath)

# DataPath = "I:/DataSets/CIBR_data/insight/*"
# DataPath = glob.glob(DataPath)

if "FL" in DataPath[0]:
    for name in DataPath:
        indx = name[-3:]
        dataPath = name + "/%s_eeg_data.npy" % indx
        labelPath = name + "/hypno_30s.csv"
        indx = int(indx)
        All_Processing(dataPath, labelPath, indx, warmingTime, Time_Sld, C3, REog, EMG, "FL")
        # All_Processing(dataPath, labelPath, indx, warmingTime, Time_Sld, C3, REog, EMG, dataBase)
        # break

if "insight" in DataPath[0]:
    for name in DataPath:
        indx = name.split("sub")[1][:2]
        dataPath = name + "/sub%s_night_eeg_data.npy" % indx
        labelPath = name + "/hypno_30s.csv"
        indx = int(indx)
        All_Processing(dataPath, labelPath, indx, warmingTime, Time_Sld, C3, REog, EMG, "IS")
        # All_Processing(dataPath, labelPath, indx, warmingTime, Time_Sld, C3, REog, EMG, dataBase)
        # break

