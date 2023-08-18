import numpy as np
import mne
import torch
import os
import yasa
import time
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")


# Function Init
def timecal(useTime):
    m, s = divmod(useTime, 60)
    h, m = divmod(m, 60)
    return "%02dh %02dm %.3fs" % (h, m, s)


def All_Processing(dataPath, indx, warmingTime, Time_Sld, C3, M2, REog, EMG, EMGREF):
    print("Processing Realtime Data, index %03d\n" % indx)
    now_init = time.time()

    # MASS channel init

    fs = 256
    model_stride = 20
    yasa_stride = 30
    realtime_stride = 3

    # Sleep Stage mapping
    mapping_yasa = {"W": 0, "N1": 0, "N2": 2, "N3": 3, "R": 4}
    mapping_model = {"W/N1": 0, "N2": 2, "N3": 3, "REM": 4}

    # Paths init
    SaveLabel = "BenchMark/RealTime_%03d" % indx
    if not os.path.exists(SaveLabel):
        os.mkdir(SaveLabel)

    # Data Loading
    print("Data loading...")
    now = time.time()
    raw = mne.io.read_raw_brainvision(dataPath, preload=True)
    raw.resample(fs)
    raw.notch_filter(50)
    raw.filter(.1, 40)
    data, _ = raw[:]
    total_length = []
    # Processor = dataProcessor(fs)
    print("Done! Use time: %s\n" % timecal(time.time() - now))

    # Predict Offline Yasa
    print("Predicting offline data by Yasa...")
    now = time.time()
    data_C3 = data[C3]
    data_REOG = data[REog]
    data_EMG = data[EMG]
    data_EMG_REF = data[EMGREF]
    data_M2 = data[M2]

    data_C3 = data_C3 - data_M2
    data_REOG = data_REOG - data_M2
    data_EMG = data_EMG - data_EMG_REF

    data_yasa = [data_C3, data_REOG, data_EMG]
    info = mne.create_info(ch_names=['C3', 'REOG', "EMG"],
                           ch_types='eeg',
                           sfreq=fs)
    raw_yasa = mne.io.RawArray(data_yasa, info, verbose=False)
    sls = yasa.SleepStaging(raw_yasa, eeg_name='C3', eog_name='REOG', emg_name="EMG")
    pred_yasa = sls.predict()
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
    data, _ = raw[:]
    data_C3 = data[C3]
    data_M2 = data[M2]
    data_REOG = data[REog]

    data_C3 = data_C3 - data_M2
    data_REOG = data_REOG - data_M2

    data_C3 = normalization(data_C3)
    data_REOG = normalization(data_REOG)
    data_my = torch.tensor([data_C3, data_REOG])
    stride = model_stride * fs
    length = len(data_C3) // stride
    for T in tqdm(range(length), desc="Data Splitting"):
        onset = stride * T
        temp_dat = data_my[:, onset: stride + onset].unsqueeze(dim=0)
        spt_data = temp_dat if T == 0 else torch.cat([spt_data, temp_dat], dim=0)
    net = SleepModel()
    # Offline Predicting
    offline_model = []
    for i in tqdm(range(len(spt_data)), desc="Offline Predicting"):
        dat = spt_data[i]
        pred_model, _ = net.predict(dat)
        for _ in range(model_stride):
            offline_model.append(mapping_model[pred_model])
    offline_model = np.array(offline_model)
    total_length.append(len(offline_model))
    print("Done! Use time: %s\n" % timecal(time.time() - now))

    data = None
    raw = None

    # Offline Data Saving
    length_min = min(total_length)
    offline_yasa = offline_yasa[:length_min]
    offline_model = offline_model[:length_min]

    np.save(SaveLabel + "/Offline_yasa.npy", offline_yasa)
    np.save(SaveLabel + "/Offline_model.npy", offline_model)

    # Realtime Predicted by Model
    print("Predicting realtime data by our model...")
    now_realtime = time.time()
    Collector_model = Collector_Model(dataPath, C3=C3, REog=REog)
    Collector_model.warmup(warmingTime)
    realtime_model = []
    usetime_model = []
    for _ in range(warmingTime):
        realtime_model.append(0)
    try:
        while True:
            now = time.time()
            dataout, (h, m, s) = Collector_model.dataout(Time_Sld, realtime_stride)
            # data_resamp = Processor.resample(dataout, 256, Time_Sld)
            # data_filt = Processor.filter(data_resamp)

            # pred, prob = net.predict(data_filt)

            pred, prob = net.predict(dataout)

            useTime = float(time.time() - now)
            usetime_model.append(useTime)
            for _ in range(realtime_stride):
                realtime_model.append(mapping_model[pred])
            print("\r", pred, prob, "Time: %.3f seconds" % useTime, "%02d:%02d:%02d" % (h, m, s), end="")


    except BaseException:
        realtime_model = np.array(realtime_model)
        usetime_model = np.array(usetime_model)
        print()
    np.save(SaveLabel + "/Realtime_model.npy", realtime_model)
    np.save(SaveLabel + "/Realtime_model_time.npy", usetime_model)
    print("Done! Use time: %s\n" % timecal(time.time() - now_realtime))

    Collector_model = None
    net = None

    # # Realtime Predicted by Yasa
    # print("Predicting realtime data by yasa...")
    # now_realtime = time.time()
    # Collector_yasa = Collector_Yasa(dataPath, C3=C3, REog=REog)
    # Collector_yasa.warmup(warmingTime)
    # realtime_yasa = []
    # usetime_yasa = []
    # for _ in range(warmingTime):
    #     realtime_yasa.append(0)
    # try:
    #     while True:
    #         now = time.time()
    #
    #         dataout, (h, m, s) = Collector_yasa.dataout(Time_Sld, realtime_stride)
    #         sls = yasa.SleepStaging(dataout, eeg_name='C3', eog_name='REOG')
    #         pred = sls.predict()
    #
    #         useTime = float(time.time() - now)
    #         print("\r", pred[-1], "Use time: %.3f seconds" % useTime, "%02d:%02d:%02d" % (h, m, s), end="")
    #         for _ in range(realtime_stride):
    #             realtime_yasa.append(mapping_yasa[pred[-1]])
    #         usetime_yasa.append(useTime)
    # except BaseException:
    #     realtime_yasa = np.array(realtime_yasa)
    #     usetime_yasa = np.array(usetime_yasa)
    #     print()
    # np.save(SaveLabel + "/Realtime_yasa.npy", realtime_yasa)
    # np.save(SaveLabel + "/Realtime_yasa_time.npy", usetime_yasa)
    # print("Done! Use time: %s\n" % timecal(time.time() - now_realtime))

    useTime_all = timecal(time.time() - now_init)

    print("Realtime data, index %03d done! Use time: %s\n" % (indx, useTime_all))


# Only Hyperparameters you will change:

# indx = 20
# dataPath = "DataStim/Real/%03d/eeg/TMR.vhdr" % indx
# dataPath = "DataStim/Real/%03d/TMR.vhdr" % indx

warmingTime = 600
Time_Sld = 20
C3 = 4
M2 = 31
REog = 19
EMG = 53
EMGREF = 56


# i = [14, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
# for indx in i:
#     dataPath = "DataStim/Real/%03d/TMR.vhdr" % indx
#     All_Processing(dataPath, indx, warmingTime, Time_Sld, C3, M2, REog, EMG, EMGREF)

indx = 14
dataPath = "DataStim/Real/%03d/TMR.vhdr" % indx
All_Processing(dataPath, indx, warmingTime, Time_Sld, C3, M2, REog, EMG, EMGREF)
