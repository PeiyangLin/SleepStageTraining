import torch
from torch import nn
import numpy as np


class Para_model:

    def __init__(self):
        fs = 256
        self.conv_fs1 = {"inchan": 3,
                         "outchan": 16,
                         "kernel": int(fs / 1),
                         "stride": int(fs / 8),
                         "padding": 0,
                         "ratio": 8}

        self.conv_fs2 = {"inchan": 3,
                         "outchan": 16,
                         "kernel": int(fs / 2),
                         "stride": int(fs / 8),
                         "padding": 0,
                         "ratio": 8}

        self.conv_fs3 = {"inchan": 3,
                         "outchan": 16,
                         "kernel": int(fs / 4),
                         "stride": int(fs / 8),
                         "padding": 0,
                         "ratio": 8}

        self.conv_n1 = {"inchan": 16,
                        "outchan": 32,
                        "kernel": 10,
                        "stride": 1,
                        "padding": 3}

        self.conv_n2 = {"inchan": 16,
                        "outchan": 32,
                        "kernel": 10,
                        "stride": 1,
                        "padding": 1}

        self.conv_n3 = {"inchan": 16,
                        "outchan": 32,
                        "kernel": 10,
                        "stride": 1,
                        "padding": 0}

        self.conv_s5 = {"inchan": 32,
                        "outchan": 32,
                        "kernel": 5,
                        "stride": 1,
                        "padding": 2}

        self.conv_pool = {"inchan": 96,
                          "outchan": 32,
                          "kernel": 7,
                          "stride": 7,
                          "padding": 0,
                          "ratio": 8}

        self.conv_s1_1 = {"inchan": 32,
                          "outchan": 64,
                          "kernel": 3,
                          "stride": 1,
                          "padding": 1}

        self.conv_s1_2 = {"inchan": 64,
                          "outchan": 64,
                          "kernel": 3,
                          "stride": 1,
                          "padding": 1}

        self.conv_s1_3 = {"inchan": 64,
                          "outchan": 32,
                          "kernel": 3,
                          "stride": 1,
                          "padding": 1}

        self.conv_pool2 = {"inchan": 32,
                           "outchan": 16,
                           "kernel": 5,
                           "stride": 5,
                           "padding": 0,
                           "ratio": 8}

        # Flatten

        self.linear = {"inchan": 208,
                       "outchan": 4}


P = Para_model()


class conv(nn.Module):

    def __init__(self, para):
        super(conv, self).__init__()
        inchan = para["inchan"]
        outchan = para["outchan"]
        kernel = para["kernel"]
        padding = para["padding"]
        stride = para["stride"]
        self.conv = nn.Conv1d(in_channels=inchan, out_channels=outchan, kernel_size=kernel, padding=padding,
                              stride=stride)
        self.bn = nn.BatchNorm1d(num_features=outchan)
        self.act = nn.ReLU(inplace=True)

    def forward(self, X):
        X = self.conv(X)
        X = self.bn(X)
        X = self.act(X)
        return X


class conv_SE(nn.Module):

    def __init__(self, para):
        super(conv_SE, self).__init__()

        inchan = para["inchan"]
        outchan = para["outchan"]
        kernel = para["kernel"]
        stride = para["stride"]
        padding = para["padding"]
        ratio = para["ratio"]

        self.conv = nn.Conv1d(in_channels=inchan, out_channels=outchan, kernel_size=kernel, padding=padding,
                              stride=stride)
        self.bn = nn.BatchNorm1d(num_features=outchan)
        self.act = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_features=inchan, out_features=outchan // ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=outchan // ratio, out_features=outchan, bias=False),
        )
        self.sig = nn.Sigmoid()
        self.outchan = outchan

        # nn.init.kaiming_normal_(self.conv.weight, nonlinearity="relu")
        nn.init.constant_(self.bn.weight, 0.5)
        nn.init.zeros_(self.bn.bias)

    def forward(self, X):
        b, c, _ = X.shape
        v = self.avgpool(X).view(b, c)
        v = self.fc(v).view(b, self.outchan, 1)

        v = self.sig(v)

        X = self.conv(X)
        X = self.bn(X)
        X = self.act(X)
        out = X * v
        return out


class conv_fs(nn.Module):

    def __init__(self, para_fs, para_norm, para_series):
        super(conv_fs, self).__init__()

        self.conv_se = conv_SE(para_fs)
        self.conv_norm = conv(para_norm)
        self.conv_seq = nn.Sequential(conv(para_series),
                                      conv(para_series),
                                      conv(para_series))

    def forward(self, X):
        X = self.conv_se(X)
        X = self.conv_norm(X)
        output = self.conv_seq(X)

        output = output + X

        return output


class Model_total(nn.Module):

    def __init__(self):
        super(Model_total, self).__init__()

        self.conv_fs1 = conv_fs(P.conv_fs1, P.conv_n1, P.conv_s5)
        self.conv_fs2 = conv_fs(P.conv_fs2, P.conv_n2, P.conv_s5)
        self.conv_fs3 = conv_fs(P.conv_fs3, P.conv_n3, P.conv_s5)

        self.conv_pool = conv_SE(P.conv_pool)

        self.conv_series = nn.Sequential(conv(P.conv_s1_1),
                                         conv(P.conv_s1_2),
                                         conv(P.conv_s1_3))

        self.conv_pool2 = conv_SE(P.conv_pool2)

        self.ft = nn.Flatten()
        self.fc = nn.Linear(P.linear["inchan"], P.linear["outchan"])

        self.softmax = nn.Softmax(dim=1)

    def forward(self, X):
        X1 = self.conv_fs1(X)
        X2 = self.conv_fs2(X)
        X3 = self.conv_fs3(X)

        X = torch.cat([X1, X2, X3], dim=1)

        X = self.conv_pool(X)

        X_s = self.conv_series(X)

        output = X + X_s

        output = self.conv_pool2(output)

        output = self.ft(output)

        output = self.fc(output)

        output = self.softmax(output)

        return output
