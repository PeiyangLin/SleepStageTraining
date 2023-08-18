import torch
import numpy as np
import pandas as pd
import glob
import time

def benchmark(label_data, pred_data):
# OFffline BenchMark
    result_matrix = np.zeros([4, 8])
    remap = [0, 1, 2, 3]
    count = 0

    for i in range(len(label_data)):
        y = remap[label_data[i]]
        pred = remap[pred_data[i]]
        result_matrix[y][pred] += 1
        if y == pred:
            count += 1
    ACC = count / len(label_data)
    ACC = round(ACC * 100, 3)

    for target in range(4):
        # Offline BenchMark
        TP = 0
        FP = 0
        TN = 0
        FN = 0

        for i in range(len(label_data)):
            y = remap[label_data[i]]
            pred = remap[pred_data[i]]
            if y == target and pred == target:
                TP += 1
            elif y != target and pred != target and y == pred:
                TN += 1
            elif y == target and pred != target:
                FN += 1
            elif y != target and pred == target:
                FP += 1

        try:
            TPR = TP / (TP + FP)
        except ZeroDivisionError:
            TPR = 0
        finally:
            result_matrix[target][4] = round(TPR * 100, 3)

        try:
            FPR = FP / (FP + TN)
        except ZeroDivisionError:
            FPR = 0
        finally:
            result_matrix[target][5] = round(FPR * 100, 3)

        try:
            RC = TP / (TP + FN)
        except ZeroDivisionError:
            RC = 0
        finally:
            result_matrix[target][6] = round(RC * 100, 3)

        try:
            F = (TPR * RC * 2) / (TPR + RC)
        except ZeroDivisionError:
            F = 0
        finally:
            result_matrix[target][7] = round(F * 100, 3)
    print("Total Accuracy:", ACC)
    r = result_matrix.T
    row_name = ["W/N1", "N2", "N3", "REM"]
    dataFrame = {"GT": row_name, "W/N1": r[0], "N2": r[1], "N3": r[2], "REM": r[3], "TPR": r[4], "FPR": r[5], "RC": r[6], "F": r[7]}
    dataFrame = pd.DataFrame(dataFrame)
    return result_matrix, dataFrame, ACC