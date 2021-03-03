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
    This is the auto parameter Bayesian search module
    requirements:
    Pytorch
    facebook/Ax (pip install ax)
    '''
    def __init__(self, path, x_train, y_train, x_val, y_val):

        '''
                 :param path: absolute path of the configuration file (.json)
                 :param x_train: raw train data dim-[N, features]
                 :param y_train: raw train label dim-[N, 1]
                 :param x_val: val data, calculate fitting loss on validation data, regrading fitting loss as model selection metric
                 :param y_val: val label
        '''
        super(AutoParameterSearch, self).__init__()

        self.parameterization = None
        self.input_dim, self.output_dim, self.num_hidden, self.hidden_dim, self.activate_func = None, None, None, None, None
        self.activate_func_list = {'ReLU': nn.ReLU(), 'Tanh': nn.Tanh(), 'Sigmoid': nn.Sigmoid()}#we can add more activate function here
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.x_train = torch.FloatTensor(x_train).to(self.device)
        self.y_train = torch.FloatTensor(y_train).to(self.device)
        self.x_val = torch.FloatTensor(x_val).to(self.device)
        self.y_val = torch.FloatTensor(y_val).to(self.device)
        self.batch_size = None
        self.fc1, self.out, self.hiddens = None, None, None
        self.path = path

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
        :param parameterization: tuned hyper-parameters
        :return: initial model
        '''
        self.input_dim = self.x_train.shape[1]
        self.output_dim = parameterization.get('output_dim', 1)
        self.num_hidden = parameterization.get('hidden_num', 1)
        self.hidden_dim = parameterization.get('hidden_dim', 50)
        try:
            self.activate_func = self.activate_func_list[parameterization.get('activate_fuc', 'ReLU')]
        except:
            self.activate_func = nn.ReLU()
            print("Please check your activation function if there is a typo. Choose among ReLU, Tanh, Sigmoid. Use ReLU to replace your choice here")
        self.fc1 = nn.Sequential(nn.Linear(in_features=self.input_dim, out_features=self.hidden_dim), self.activate_func)
        self.out = nn.Linear(self.hidden_dim, self.output_dim)

        if self.num_hidden-1: ## Here we check if we want to consider any hidden alyer or not if yes it will be added to the layer one by one
            Layers = []
            for i in range(self.num_hidden-1):
                Layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
                Layers.append(self.activate_func)
            self.hiddens = nn.Sequential(*Layers)
            model = nn.Sequential(self.fc1, self.hiddens, self.out)
        else:
            model = nn.Sequential(self.fc1, self.out)

        return model

    def shuffle_data(self):
        '''
            Shuffle training data
            :return: shuffled data
        '''
        indices = np.arange(self.x_train.shape[0])
        random.shuffle(indices)
        return self.x_train[indices], self.y_train[indices]

    def train_net(self, model, parameterization):
        '''
        Train current model
        :param model: initial model
        :param parameterization: tuned hyper-parameters
        :return: trained model, fitting loss of current hyper-parameter configuration
        '''
        model = model.to(self.device) ## This is for moving the whole model to a chosen device (CPU, GPU)

        criterion = nn.MSELoss() ## selecting a proper loss function
        optimizer = torch.optim.SGD(model.parameters(), lr=parameterization.get('lr', 0.001), momentum=parameterization.get('momentum', 0.9))
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=parameterization.get("step_size", 30),
                                                    gamma=parameterization.get('gamma', 1.0)) ## This will 
        num_epochs = parameterization.get("num_epochs", 1000)
        for _ in range(num_epochs):
            x, y = self.shuffle_data()
            for i in range(self.x_train.shape[0]//self.batch_size):
                optimizer.zero_grad()
                outputs = model(x[i*self.batch_size: (i+1)*self.batch_size])
                loss = criterion(outputs, y[i*self.batch_size: (i+1)*self.batch_size])
                loss.backward()
                optimizer.step()
                scheduler.step()

        with torch.no_grad():
            Y = model(self.x_val)
            loss_ = criterion(Y, self.y_val)
            # Some parameters will cause the gradient of the model to become nan during training, 
            # and the loss will also become nan, which affects the model selection. This is to avoid the search being interrupted. 
            if np.isnan(loss_.item()):
                return model, 10000.0
            else:
                return model, loss_.item()


    def evaluate_net(self, parameterization):
        '''
        Evaluate the current model based on fitting loss on dataset
        :param parameterzation: tuned hyper-parameters
        :return: evaluation metrics-fitting loss
        '''
        print(parameterization)
        self.batch_size = parameterization.get('batch_size', self.x_train.shape[0])
        model = self.init_net(parameterization)
        model, loss = self.train_net(model, parameterization)

        return loss

    def load_para(self):
        '''
        Load hyper-parameter configuration file (json)
        :return: None
        '''
        if self.path.endswith('.json'):
            with open(self.path, 'r', encoding='utf8') as load_f:
                load_dict = json.load(load_f)
                self.parameterization = load_dict
        else:
            print('Please use json file.')

    def optimize_para(self, total_trials = 15, visualize = True):
        '''
        Find best parameters based on Bayesian Optimization. Store optimal model hyper-parameters and parameters.
        :return: Trained Optimal Model
        '''
        self.load_para()
        best_parameters, values, experiment, model = optimize(
            parameters=self.parameterization,
            evaluation_function=self.evaluate_net,
            objective_name='loss',
            minimize=True,
            total_trials=total_trials,
            random_seed=0)
        np.save('Optimal_HyperParameters.npy', best_parameters)
        optimal_model = self.init_net(best_parameters)
        optimal_model, _ = self.train_net(optimal_model, best_parameters)
        torch.save(optimal_model.state_dict(), "Optimal_Model_Parameter.pth")

        if visualize:
            with torch.no_grad():
                y_pred = optimal_model(self.x_val)
                y_pred = y_pred.cpu().detach().numpy()
                x_data = self.x_val.cpu().detach().numpy()
                y_data = self.y_val.cpu().detach().numpy()
                plt.cla()
                plt.scatter(x_data, y_data)
                plt.plot(x_data, y_pred, color = 'r', lw=3)
                plt.show()

        return optimal_model

# %%
