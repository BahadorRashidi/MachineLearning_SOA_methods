#%%
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import json
import time
import random

#%%
class AutoParameterSearch(nn.Module):
    '''
    This class tends to fit a regression parameter into the given dataset
    requirements:
    Pytorch, numpy
    '''
    def __init__(self, path, x_train, y_train, x_val, y_val):

        '''
            :param path: absolute path of the configuration file (.json)
            :param x_train: raw train data dim-[N, features]
            :param y_train: raw train label dim-[N, 1]
            :param x_val: val data, Validation input data for calculating the fitting loss and compare the accuracy according to it
            :param y_val: val label, Validation output data for calculating the fitting loss and compare the accuracy according to it
        '''
        super(AutoParameterSearch, self).__init__()

        self.parameterization = None
        self.input_dim, self.output_dim, self.num_hidden, self.hidden_dim, self.activate_func = None, None, None, None, None
        self.activate_func_list = {'ReLU': nn.ReLU(), 'Tanh': nn.Tanh(), 'Sigmoid': nn.Sigmoid()} # we can add more activate function here
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.x_train = torch.FloatTensor(x_train).to(self.device)
        self.y_train = torch.FloatTensor(y_train).to(self.device)
        self.x_val = torch.FloatTensor(x_val).to(self.device)
        self.y_val = torch.FloatTensor(y_val).to(self.device)
        self.batch_size = None
        self.fc1, self.out, self.hiddens = None, None, None
        self.path = path

    '''
    Forward propagation function for the trained DNN.
    This function checks if there is any given hidden layer considered, otherwis eit will only consider the fc1 
    '''
    def forward(self, X):
        '''
            :param X: Input of the NN dim-[N, features]
            :return: Output of the NN dim-[N, 1]
        '''
        X = self.fc1(X)
        if not self.hiddens:
            X = self.hiddens(X)
        X = self.out(X)
        return X

    def init_net(self, parameterization):
        '''
            :param parameterization: tuned hyper-parameters which is a python dicttioanry
            :return: initial model
        '''
        self.input_dim = parameterization.get('input_dim', self.x_train.shape[1])
        self.output_dim = parameterization.get('output_dim', 1)
        self.num_hidden = parameterization.get('hidden_num', 1)
        self.hidden_dim = parameterization.get('hidden_dim', 500)
        try:
            self.activate_func = self.activate_func_list[parameterization.get('activate_fuc', 'ReLU')]
        except:
            self.activate_func = nn.ReLU()
            print("Please check your activation function if there is a typo. Choose among ReLU, Tanh, Sigmoid. Use ReLU to replace your choice here")
        self.fc1 = nn.Sequential(nn.Linear(in_features=self.input_dim, out_features=self.hidden_dim),
                                 self.activate_func)
        self.out = nn.Linear(self.hidden_dim, self.output_dim)

        # self.hiddens = None
        if self.num_hidden-1:
            Layers = []
            for i in range(self.num_hidden-1):
                Layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
                Layers.append(self.activate_func)
            self.hiddens = nn.Sequential(*Layers)
            model = nn.Sequential(self.fc1, self.hiddens, self.out)
        else:
            model = nn.Sequential(self.fc1, self.out)

        return model







