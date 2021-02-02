import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Beagle(nn.Module):
    """
    Neural net models over genomic sequence.
    Input:
        - sequence_length: int (default 1000) 
        - Shape: (N, 5, sequence_length, 1) with batch size N.
    
    Output:
        - prediction (Tensor): float torch tensor of shape (N, )
    
    TODO: Finish docstring.
    """
    def __init__(self, args):
        """
        Parameters
        ----------
        sequence_length : int
        n_genomic_features : int
        """
        super(Beagle, self).__init__()

        self.dropout = args.dropout
        self.num_cell_types = 1
        self.conv1 = nn.Conv2d(5, 300, (19, 1), stride = (1, 1), padding=(9,0))
        self.conv2 = nn.Conv2d(300, 200, (11, 1), stride = (1, 1), padding = (5,0))
        self.conv3 = nn.Conv2d(200, 200, (7, 1), stride = (1, 1), padding = (4,0))
        self.bn1 = nn.BatchNorm2d(300)
        self.bn2 = nn.BatchNorm2d(200)
        self.bn3 = nn.BatchNorm2d(200)
        self.maxpool1 = nn.MaxPool2d((3, 1))
        self.maxpool2 = nn.MaxPool2d((4, 1))
        self.maxpool3 = nn.MaxPool2d((4, 1))

        self.fc1 = nn.Linear(4200, 1000)
        self.bn4 = nn.BatchNorm1d(1000)

        self.fc2 = nn.Linear(1000, 1000)
        self.bn5 = nn.BatchNorm1d(1000)

        self.fc3 = nn.Linear(1000, self.num_cell_types)

    def forward(self, s):
        s = s.permute(0, 2, 1).contiguous()                          # batch_size x 5 x 1000
        s = s.view(-1, 5, 1000, 1)                                   # batch_size x 5 x 1000 x 1 [5 channels]
        s = self.maxpool1(F.relu(self.bn1(self.conv1(s))))           # batch_size x 300 x 333 x 1
        s = self.maxpool2(F.relu(self.bn2(self.conv2(s))))           # batch_size x 200 x 83 x 1
        s = self.maxpool3(F.relu(self.bn3(self.conv3(s))))           # batch_size x 200 x 21 x 1
        s = s.view(-1, 4200)
        conv_out = s

        s = F.dropout(F.relu(self.bn4(self.fc1(s))), p=self.dropout, training=self.training)  # batch_size x 1000
        s = F.dropout(F.relu(self.bn5(self.fc2(s))), p=self.dropout, training=self.training)  # batch_size x 1000
        
        s = self.fc3(s)

        return s, conv_out



#class MLP(nn.Module):
#    """Just an MLP"""
#    def __init__(self, n_inputs, n_outputs, width, depth, drop_out):
#        super(MLP, self).__init__()
#
#        self.input = nn.Linear(n_inputs, width)
#        self.dropout = nn.Dropout(dropout)
#        self.hiddens = nn.ModuleList([
#            nn.Linear(width,width)
#            for _ in range(depth-2)])
#        self.output = nn.Linear(width, n_outputs)
#        self.n_outputs = n_outputs
#
#    def forward(self, x):
#        x = self.input(x)
#        x = self.dropout(x)
#        x = F.relu(x)
#        for hidden in self.hiddens:
#            x = hidden(x)
#            x = self.dropout(x)
#            x = F.relu(x)
#        x = self.output(x)
#        return x


"""
DeepSEA architecture (Zhou & Troyanskaya, 2015).
Based on https://github.com/FunctionLab/selene/blob/master/models/deepsea.py
"""

class DeepSEA(nn.Module):
    def __init__(self, sequence_length, n_genomic_features):
        """
        Parameters
        ----------
        sequence_length : int
        n_genomic_features : int
        """
        super(DeepSEA, self).__init__()
        conv_kernel_size = 8
        pool_kernel_size = 4

        self.conv_net = nn.Sequential(
            nn.Conv1d(4, 320, kernel_size=conv_kernel_size),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(
                kernel_size=pool_kernel_size, stride=pool_kernel_size),
            nn.Dropout(p=0.2),

            nn.Conv1d(320, 480, kernel_size=conv_kernel_size),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(
                kernel_size=pool_kernel_size, stride=pool_kernel_size),
            nn.Dropout(p=0.2),

            nn.Conv1d(480, 960, kernel_size=conv_kernel_size),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5))

        reduce_by = conv_kernel_size - 1
        pool_kernel_size = float(pool_kernel_size)
        self.n_channels = int(
            np.floor(
                (np.floor(
                    (sequence_length - reduce_by) / pool_kernel_size)
                 - reduce_by) / pool_kernel_size)
            - reduce_by)
        self.classifier = nn.Sequential(
            nn.Linear(960 * self.n_channels, n_genomic_features),
            nn.ReLU(inplace=True),
            nn.Linear(n_genomic_features, n_genomic_features),
            nn.Sigmoid())

    def forward(self, x):
        """Forward propagation of a batch.
        """
        out = self.conv_net(x)
        reshape_out = out.view(out.size(0), 960 * self.n_channels)
        predict = self.classifier(reshape_out)
        return predict

"""
def criterion():
    return nn.BCELoss()

def get_optimizer(lr):
    # The optimizer and the parameters with which to initialize the optimizer. At a later time, we initialize the optimizer by also passing in the model parameters (`model.parameters()`). We cannot initialize the optimizer until the model has been initialized.
    return (torch.optim.SGD, {"lr": lr, "weight_decay": 1e-6, "momentum": 0.9})
"""



"""
DanQ architecture (Quang & Xie, 2016).
"""

class DanQ(nn.Module):
    def __init__(self, sequence_length, n_genomic_features):
        """
        Parameters
        ----------
        sequence_length : int
            Input sequence length
        n_genomic_features : int
            Total number of features to predict
        """
        super(DanQ, self).__init__()
        self.nnet = nn.Sequential(
            nn.Conv1d(4, 320, kernel_size=26),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(
                kernel_size=13, stride=13),
            nn.Dropout(0.2))

        self.bdlstm = nn.Sequential(nn.LSTM(320, 320, num_layers=1, batch_first=True, bidirectional=True))

        self._n_channels = math.floor(
            (sequence_length - 25) / 13)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self._n_channels * 640, 925),
            nn.ReLU(inplace=True),
            nn.Linear(925, n_genomic_features),
            nn.Sigmoid())

    def forward(self, x):
        """Forward propagation of a batch.
        """
        out = self.nnet(x)
        reshape_out = out.transpose(0, 1).transpose(0, 2)
        out, _ = self.bdlstm(reshape_out)
        out = out.transpose(0, 1)
        reshape_out = out.contiguous().view(
            out.size(0), 640 * self._n_channels)
        predict = self.classifier(reshape_out)
        return predict

"""
def criterion():
    return nn.BCELoss()

def get_optimizer(lr):
    return (torch.optim.RMSprop, {"lr": lr})
"""