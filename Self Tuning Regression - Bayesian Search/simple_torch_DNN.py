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
import math
import pandas as pd
#%% 
'''
Create a sample data set here for testing your model
Note: uncomment if it is the first time you run this code
'''
# num_samples = 5000
# X = np.zeros((num_samples,4))
# X[:,0:2] = np.random.rand(num_samples,2) # generate uniformly distributed data
# deterministic_trend = np.linspace(0,5,num_samples)
# X[:,0] = 0.2*X[:,0] + deterministic_trend
# X[:,2] = np.exp(-15 + 2*X[:,0]) 
# X[:,3] = np.linspace(-2,2,num_samples)*np.sin(np.linspace(-2,2,num_samples))
# Y = X[:,0]*X[:,3] + X[:,1]*X[:,2] + 0.05*np.random.randn(num_samples)
# plt.plot(X)
# plt.show()
# plt.plot(Y)
# data_dict = {'x1': X[:,0], 'x2': X[:,1],'x3': X[:,2],'x4': X[:,3], 'Y': Y}
# df = pd.DataFrame(data_dict)
# df.to_csv('data.csv', index=False)

#%%
class simple_torch_DNN(nn.Module):
    '''
        This is a simpel DNN regression with given structure
    '''
    def __init__(self, x_train, y_train, x_val, y_val, parameters):

        '''
                 :param x_train: raw train data dim-[N, features]
                 :param y_train: raw train label dim-[N, 1]
                 :param x_val: val data, calculate fitting loss on validation data, regrading fitting loss as model selection metric
                 :param y_val: val label
        '''
        super(simple_torch_DNN, self).__init__()

        self.input_dim, self.output_dim, self.num_hidden, self.hidden_dim, self.activate_func = None, None, None, None, None
        self.activate_func_list = {'ReLU': nn.ReLU(), 'Tanh': nn.Tanh(), 'Sigmoid': nn.Sigmoid()}#we can add more activate function here
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.x_train = torch.FloatTensor(x_train).to(self.device)
        self.y_train = torch.FloatTensor(y_train).to(self.device)
        self.x_val = torch.FloatTensor(x_val).to(self.device)
        self.y_val = torch.FloatTensor(y_val).to(self.device)
        self.batch_size = None
        self.func1 = None
        self.out = None
        self.hiddens = None
        self.path = path
        self.parameterization = parameters

    def forward(self, X):
        X = self.func1(X)
        if not self.hiddens:
            X = self.hiddens(X)
        X = self.out(X)
        return X

    def init_net(self):
        '''
        :param parameterization: tuned hyper-parameters
        :return: initial model
        '''
        self.input_dim = self.x_train.shape[1]
        self.output_dim = self.y_train.shape[1]

        try:
            self.activate_func = self.activate_func_list[self.parameterization.get('activate_fuc', 'ReLU')]
        except:
            self.activate_func = nn.ReLU()
            print("Please check your activation function if there is a typo. Choose among ReLU, Tanh, Sigmoid. Use ReLU to replace your choice here")


        try: 
            num_hidden_layers = len(self.parameterization['DNN_hidden_layer_Structure'])
            DNN_hidden_layer_Structure = self.parameterization['DNN_hidden_layer_Structure']
            ## Here we determine the structure of the DNN model

            if num_hidden_layers == 1:    
                self.fc1 = nn.Sequential(nn.Linear(in_features=self.input_dim, out_features=DNN_hidden_layer_Structure[0]), self.activate_func)
                self.out = nn.Linear(DNN_hidden_layer_Structure[0], self.output_dim)
                model = nn.Sequential(self.fc1, self.out)
                return model

            elif num_hidden_layers > 1: 
                self.fc1 = nn.Sequential(nn.Linear(in_features=self.input_dim, out_features=DNN_hidden_layer_Structure[0]), self.activate_func)
                self.out = nn.Linear(DNN_hidden_layer_Structure[0], self.output_dim)
                Layers = []
                for i in range(num_hidden_layers - 1):
                    Layers.append(nn.Linear(DNN_hidden_layer_Structure[i+1], DNN_hidden_layer_Structure[i+2]))
                    Layers.append(self.activate_func)
                self.hiddens = nn.Sequential(*Layers)
                model = nn.Sequential(self.fc1, self.hiddens, self.out)
                return model
        except:
            print('DNN structure is not provided and a default layer is considered!')
            self.fc1 = nn.Sequential(nn.Linear(in_features=self.input_dim, out_features=self.hidden_dim), self.activate_func)
            self.out = nn.Linear(self.hidden_dim, self.output_dim)
            model = nn.Sequential(self.fc1, self.out)
            return model

    def shuffle_data(self):
        '''
        This function generate an index array and randomly shuffle it
        '''
        indices = np.arange(self.x_train.shape[0])
        random.shuffle(indices)
        return self.x_train[indices], self.y_train[indices] 

    def train_model(self, model):

        # assign the model to the device
        model = model.to(self.device)
        loss_function = nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=self.parameterization.get('learning_rate',1e-2),
                                    momentum=parameterization.get('momentum', 0.9))

        num_epochs = self.parameterization.get('epochs', 500)
        for i in range(num_epochs):
            # x, y = self.shuffle_data() ## This will shuffel the training data for each epoch
            for i in range(self.x_train.shape[0]//self.parameterization.get('batch_size', 64)):
                y_pred = model(self.x_train[i*self.parameterization.get('batch_size', 64):(i+1)*self.parameterization.get('batch_size', 64)])
                optimizer.zer_grad()
                loss = loss_function(y_pred, self.y_train[i*self.parameterization.get('batch_size', 64):(i+1)*self.parameterization.get('batch_size', 64)])
                loss.backward()
                optimizer.step()

        with torch.no_grad():
            y_val_pred = model(self.y_val)
            loss_ = loss_function(y_val_pred, self.y_val)
            if np.isnan(loss_.item()):
                print('Your parameters does not make sense and resulted in Nan!!')
                return model, 10000.0
            else:
                return model, loss_.item()


    def fit_model(self, model, visualize = True):
        model = self.init_net()
        model, loss = self.train_net(model)

        if visualize:
            with torch.no_grad():
                y_pred = model(self.x_val)
                y_pred = y_pred.cpu().detach().numpy()
                x_data = self.x_val.cpu().detach().numpy()
                y_data = self.y_val.cpu().detach().numpy()
                plt.cla()
                plt.scatter(x_data, y_data)
                plt.plot(x_data, y_pred, color = 'r', lw=3)
                plt.show()
        return loss







        
        

        






#%%
parameters = {
    'DNN_hidden_layer_Structure' : [32,64,32],
    'activate_fuc' : 'ReLU',


}





#%%




