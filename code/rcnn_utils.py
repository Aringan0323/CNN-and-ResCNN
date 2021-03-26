import numpy as np
import pickle as pkl
import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(0)

# data loader function, return four datasets: train_data_x, train_data_y, test_data_x, test_data_y
def load_dataset():
    train_data_x = pkl.load(open('../datasets/train_data_x.pkl', 'rb'), encoding='latin1')
    train_data_y = pkl.load(open('../datasets/train_data_y.pkl', 'rb'), encoding='latin1')
    test_data_x = pkl.load(open('../datasets/test_data_x.pkl', 'rb'), encoding='latin1')
    test_data_y = pkl.load(open('../datasets/test_data_y.pkl', 'rb'), encoding='latin1')
    train_data_x = torch.from_numpy(np.transpose(train_data_x, (0,3,1,2)))
    test_data_x = torch.from_numpy(np.transpose(test_data_x, (0,3,1,2)))
    train_data_y = torch.from_numpy(np.transpose(train_data_y, (1,0)))
    test_data_y = torch.from_numpy(np.transpose(test_data_y, (1,0)))
    return (train_data_x, train_data_y, test_data_x, test_data_y)


# RCNN model
class Net(nn.Module):

    def __init__(self, dropout=False):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 6, 1)
        self.conv3 = nn.Conv2d(6, 6, 1)
        self.conv4 = nn.Conv2d(6, 12, 5)
        self.pool = nn.MaxPool2d(2,2)
        if dropout:
            self.dropout = nn.Dropout(p=0.2)
        self.has_dropout = dropout
        self.lin1 = nn.Linear(2028, 120)
        self.lin2 = nn.Linear(120, 64)
        self.lin3 = nn.Linear(64, 2)

    def forward(self, x):
        relu = nn.ReLU()
        sigmoid = nn.Sigmoid()

        x = relu(self.conv1(x))

        x = self.pool(x)

        resc1 = relu(self.conv2(x))

        resc2 = relu(self.conv3(resc1))
        x = relu(torch.add(x, resc2))
        x = self.pool(self.conv4(x)).reshape(-1, 2028)
        if self.has_dropout:
            x = self.dropout(x)
        x = relu(self.lin1(x))
        x = sigmoid(self.lin2(x))
        x = self.lin3(x)
        return x
