import random
import numpy as np 
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score
import torch.nn.functional as F
import logging
from time import time
import params
import utils
from ENFN import FN
from pretrained_CNN import DenseNet121 as pretrained_DenseNet121 
from pretrained_CNN import pretrainedResnetModel
from torchvision.models import resnet18, resnet101
from torchvision.utils import make_grid

class MLP(nn.Module):

    def __init__(self, batch_size, device, hyperparams_dict, parameters, drop_prob = 0.5):
        super(MLP, self).__init__()
        
        self.batch_size = batch_size
        self.device = device
        
        self.hidden_dim = hyperparams_dict['hidden dim']
        self.only_tv = hyperparams_dict['tv only']
        self.task = hyperparams_dict['task']

        self.in_seq_len = parameters.IN_SEQ_LEN
        # Initial Convs
        if self.only_tv:
            self.input_dim = 2
        else:
            self.input_dim = 18
        
        self.dropout = nn.Dropout(drop_prob)

        if self.task == params.CLASSIFICATION or self.task == params.DUAL:
                self.fc1 = nn.Linear(self.input_dim*self.in_seq_len, self.hidden_dim)
                self.fc2 = nn.Linear(self.hidden_dim,3)
            
        if self.task == params.REGRESSION or self.task == params.DUAL:
            self.fc1_ttlc = nn.Linear(self.input_dim*self.in_seq_len, 512)
            self.fc2_ttlc = nn.Linear(512,1)

    def lc_forward(self,x_in):
        
        x_in = x_in[0]
        if self.only_tv:
            x_in = x_in[:,:,:2]
        x = x_in.view(self.batch_size, self.in_seq_len * self.input_dim)
        h1 = F.relu(self.fc1(x))
        out = self.dropout(h1)
        out = self.fc2(out)
        return out

    def ttlc_forward(self,x_in):
        
        x_in = x_in[0]
        if self.only_tv:
            x_in = x_in[:,:,:2]
        x = x_in.view(self.batch_size, self.in_seq_len * self.input_dim)
        h1 = F.relu(self.fc1_ttlc(x))
        out = self.dropout(h1)
        out = self.fc2_ttlc(out)

        return out
    
    def forward(self, x_in):
        features = 0
        if self.task == params.CLASSIFICATION or self.task == params.DUAL:
            lc_pred = self.lc_forward(x_in)
        else:
            lc_pred = 0
        
        if self.task == params.REGRESSION or self.task == params.DUAL:
            ttlc_pred = self.ttlc_forward(x_in)
        else:
            ttlc_pred = 0
        
        return {'lc_pred':lc_pred, 'ttlc_pred':ttlc_pred, 'features': features}

        


class VanillaLSTM(nn.Module):

    def __init__(self, batch_size, device, hyperparams_dict, parameters, drop_prob = 0.5):
        super(VanillaLSTM, self).__init__()
        
        self.batch_size = batch_size
        self.device = device
        
        self.hidden_dim = hyperparams_dict['hidden dim']
        self.num_layers = hyperparams_dict['layer number']
        self.only_tv = hyperparams_dict['tv only']
        self.task = hyperparams_dict['task']

        self.in_seq_len = parameters.IN_SEQ_LEN
        # Initial Convs
        if self.only_tv:
            self.input_dim = 2
        else:
            self.input_dim = 18

        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers, batch_first = True, dropout= drop_prob)
        self.dropout = nn.Dropout(drop_prob)
        # Define the output layer
        
        if self.task == params.CLASSIFICATION or self.task == params.DUAL:
            self.fc1 = nn.Linear(self.hidden_dim, 128)
            self.fc2 = nn.Linear(128,3)
        
        if self.task == params.REGRESSION or self.task == params.DUAL:
            self.fc1_ttlc = nn.Linear(self.hidden_dim, 512)
            self.fc2_ttlc = nn.Linear(512,1)
        
        

    def init_hidden(self):
        # This is what we'll initialise our hidden state as
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))

    def lstm_forward(self, x_in):
        # Forward pass through LSTM layer
        # shape of lstm_out: [batch_size, seq_len, hidden_dim]
        # shape of self.hidden: (a, b), where a and b both 
        # have shape (num_layers, batch_size, hidden_dim).
        x_in = x_in[0]
        if self.only_tv:
            x_in = x_in[:,:,:2]
        x = x_in.view(self.batch_size, self.in_seq_len, self.input_dim)
        lstm_out, self.hidden = self.lstm(x)
        
        # Only take the output from the final timetep
        # Can pass on the entirety of lstm_out to the next layer if it is a seq2seq prediction
        lstm_out = lstm_out.transpose(0,1)
        out = self.dropout(lstm_out[-1].view(self.batch_size, -1))
        return out
    
    def ttlc_forward(self, x):
        x = self.dropout(x)
        out = F.relu(self.fc1_ttlc(x))
        out = self.dropout(out)
        out = F.relu(self.fc2_ttlc(out))
        return out
    
    def lc_forward(self, x):
        x = self.dropout(x)
        out = F.relu(self.fc1(x))
        out = self.dropout(out)
        out = self.fc2(out)
        return out

    def forward(self,x_in):
        lstm_out = self.lstm_forward(x_in)
        
        if self.task == params.CLASSIFICATION or self.task == params.DUAL:
            lc_pred = self.lc_forward(lstm_out)
        else:
            lc_pred = 0
        
        if self.task == params.REGRESSION or self.task == params.DUAL:
            ttlc_pred = self.ttlc_forward(lstm_out)
        else:
            ttlc_pred = 0
        
        return {'lc_pred':lc_pred, 'ttlc_pred':ttlc_pred, 'features': lstm_out}
        
class VanillaGRU(nn.Module):

    def __init__(self, batch_size, device, hyperparams_dict, parameters, drop_prob = 0.5):
        super(VanillaGRU, self).__init__()
        
        self.batch_size = batch_size
        self.device = device
        
        self.hidden_dim = hyperparams_dict['hidden dim']
        self.num_layers = hyperparams_dict['layer number']
        self.only_tv = hyperparams_dict['tv only']
        self.task = hyperparams_dict['task']

        self.in_seq_len = parameters.IN_SEQ_LEN
        # Initial Convs
        if self.only_tv:
            self.input_dim = 2
        else:
            self.input_dim = 18

        self.gru = nn.GRU(self.input_dim, self.hidden_dim, self.num_layers, batch_first = True, dropout= drop_prob)
        self.dropout = nn.Dropout(drop_prob)
        # Define the output layer
        
        if self.task == params.CLASSIFICATION or self.task == params.DUAL:
            self.fc1 = nn.Linear(self.hidden_dim, 128)
            self.fc2 = nn.Linear(128,3)
        
        if self.task == params.REGRESSION or self.task == params.DUAL:
            self.fc1_ttlc = nn.Linear(self.hidden_dim, 512)
            self.fc2_ttlc = nn.Linear(512,1)
        
        

    def init_hidden(self):
        # This is what we'll initialise our hidden state as
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))

    def gru_forward(self, x_in):
        # Forward pass through LSTM layer
        # shape of lstm_out: [batch_size, seq_len, hidden_dim]
        # shape of self.hidden: (a, b), where a and b both 
        # have shape (num_layers, batch_size, hidden_dim).
        x_in = x_in[0]
        if self.only_tv:
            x_in = x_in[:,:,:2]
        x = x_in.view(self.batch_size, self.in_seq_len, self.input_dim)
        gru_out, self.hidden = self.gru(x)
        
        # Only take the output from the final timetep
        # Can pass on the entirety of lstm_out to the next layer if it is a seq2seq prediction
        gru_out = gru_out.transpose(0,1)
        out = self.dropout(gru_out[-1].view(self.batch_size, -1))
        return out
    
    def ttlc_forward(self, x):
        x = self.dropout(x)
        out = F.relu(self.fc1_ttlc(x))
        out = self.dropout(out)
        out = F.relu(self.fc2_ttlc(out))
        return out
    
    def lc_forward(self, x):
        x = self.dropout(x)
        out = F.relu(self.fc1(x))
        out = self.dropout(out)
        out = self.fc2(out)
        return out

    def forward(self,x_in):
        gru_out = self.gru_forward(x_in)
        
        if self.task == params.CLASSIFICATION or self.task == params.DUAL:
            lc_pred = self.lc_forward(gru_out)
        else:
            lc_pred = 0
        
        if self.task == params.REGRESSION or self.task == params.DUAL:
            ttlc_pred = self.ttlc_forward(gru_out)
        else:
            ttlc_pred = 0
        
        return {'lc_pred':lc_pred, 'ttlc_pred':ttlc_pred, 'features': gru_out}
        

class VanillaCNN(nn.Module):
    def __init__(self, batch_size, device, hyperparams_dict, parameters, drop_prob = 0.5):
        super(VanillaCNN, self).__init__()
        self.batch_size = batch_size
        self.device = device
        self.resume = parameters.resume
        # Hyperparams:
        self.num_channels = hyperparams_dict['channel number']
        self.kernel_size = hyperparams_dict['kernel size']
        self.single_merged_ch = hyperparams_dict['merge channels']
        self.task = hyperparams_dict['task']
        # self.probabilistic_model = hyperparams_dict['probabilistic model']
        # self.LSTM_model = hyperparams_dict['LSTM model']
        self.padding = int((self.kernel_size -1)/2) 
        self.lr = parameters.LR 
        self.SEQ_LEN = parameters.SEQ_LEN
        self.in_seq_len = parameters.IN_SEQ_LEN
        self.image_height = parameters.IMAGE_HEIGHT
        self.image_width = parameters.IMAGE_WIDTH 
        self.vector_field_available = parameters.vector_field_available
        self.TrainThePretrained = parameters.TrainThePretrained

        
        self.output_horizon = int((self.SEQ_LEN-self.in_seq_len+1)/parameters.RES)
        self.input_horizon = parameters.INPUT_HORIZON
        self.model_type = hyperparams_dict['model type']



        # Initial Convs
        if self.vector_field_available:
            self.in_channel = 4 * parameters.INPUT_HORIZON
        else:
            self.in_channel = 2 * parameters.INPUT_HORIZON

        self.dropout = nn.Dropout(drop_prob)
        if (self.model_type == 'Resnet_LSTM'):
            self.resnet = resnet18(pretrained=self.TrainThePretrained)
            # self.resnet_fc = nn.Linear(1000, 256)

            self.LSTM_ttlc = nn.GRU(1000, 256, 2, batch_first = True)
            self.LSTM_fc = nn.Linear(256, 64)
            self.LSTM_fc_x = nn.Linear(64, self.output_horizon)
            self.LSTM_fc_y = nn.Linear(64, self.output_horizon)

        elif self.model_type == 'CNN-Linear':
            self.init_conv1 = nn.Conv2d(int(self.in_channel/self.input_horizon), self.num_channels, kernel_size=self.kernel_size, stride=1, padding = self.padding)  # (100,75)
            self.init_pool2 = nn.MaxPool2d(2, padding = 1) 
            self.init_conv3 = nn.Conv2d(self.num_channels, self.num_channels, kernel_size=self.kernel_size, stride=1, padding = self.padding) # (51,38)
            self.init_pool4 = nn.MaxPool2d(2)
            self.init_conv5 = nn.Conv2d(self.num_channels, self.num_channels, kernel_size=self.kernel_size, stride=1, padding = self.padding) # (25, 19)
            self.init_pool6 = nn.MaxPool2d(2, padding = 1)        

            self.fc1_ttlc = nn.Linear(16*45*34, 512) # 16*5*33 for not grid, 16*45*34 for gird
            self.fc2_ttlc_x = nn.Linear(512,self.output_horizon)#output_horizon size
            self.fc2_ttlc_y = nn.Linear(512,self.output_horizon)

        elif self.model_type == 'probabilistic':
            self.init_conv1 = nn.Conv2d(self.in_channel, self.num_channels, kernel_size=self.kernel_size, stride=1, padding = self.padding)  # (100,75)
            self.init_pool2 = nn.MaxPool2d(2, padding = 1) 
            self.init_conv3 = nn.Conv2d(self.num_channels, self.num_channels, kernel_size=self.kernel_size, stride=1, padding = self.padding) # (51,38)
            self.init_pool4 = nn.MaxPool2d(2)
            self.init_conv5 = nn.Conv2d(self.num_channels, self.num_channels, kernel_size=self.kernel_size, stride=1, padding = self.padding) # (25, 19)
            self.init_pool6 = nn.MaxPool2d(2, padding = 1)        

            self.fc1_ttlc = nn.Linear(16*5*33, 512) # 16*5*33
            self.fc2_ttlc_muX = nn.Linear(512,self.output_horizon)#output_horizon size
            self.fc2_ttlc_muY = nn.Linear(512,self.output_horizon)#output_horizon size
            self.fc2_ttlc_sigX = nn.Linear(512,self.output_horizon)#output_horizon size
            self.fc2_ttlc_sigY = nn.Linear(512,self.output_horizon)#output_horizon size
            self.fc2_ttlc_rho = nn.Linear(512,self.output_horizon)#output_horizon size
        
        elif self.model_type == 'pretrained_denseNet':
            self.feat_extract = pretrained_DenseNet121(pretrained=self.TrainThePretrained)
            self.fc2_ttlc_x = nn.Linear(1024,self.output_horizon)#output_horizon size
            self.fc2_ttlc_y = nn.Linear(1024,self.output_horizon)

        elif self.model_type == 'pretrainedResnetModel':
            self.feat_extract = pretrainedResnetModel(pretrained=self.TrainThePretrained, resnet=18)
            self.fc2_ttlc_x = nn.Linear(1000,self.output_horizon)#output_horizon size
            self.fc2_ttlc_y = nn.Linear(1000,self.output_horizon)
   
        elif self.model_type == '3DCNN':
            if(self.vector_field_available):
                self.conv_occ_mask_layer1 = self._3D_conv_layer_set(1, 16)
                self.conv_vx_layer1 = self._3D_conv_layer_set(1, 16)
                self.conv_vy_layer1 = self._3D_conv_layer_set(1, 16)
                self.conv_occ_mask_layer2 = self._3D_conv_layer_set(16, 32)
                self.conv_vx_layer2 = self._3D_conv_layer_set(16, 32)
                self.conv_vy_layer2 = self._3D_conv_layer_set(16, 32)
                self.fc = nn.Linear((3*32*1*6*62), 512)
                self.relu = nn.LeakyReLU()
                self.batch=nn.BatchNorm1d(512)
                self.drop=nn.Dropout(p=0.2) 
                self.fc_x_1 = nn.Linear(512, 128)
                self.fc_x_2 = nn.Linear(128, self.output_horizon)
                self.fc_y_1 = nn.Linear(512, 128)
                self.fc_y_2 = nn.Linear(128, self.output_horizon)
            
            else:
                self.conv_occ_mask_layer1 = self._3D_conv_layer_set(1, 16)
                self.conv_occ_mask_layer2 = self._3D_conv_layer_set(16, 32)                
                self.fc = nn.Linear((32*1*6*62), 256)
                self.relu = nn.LeakyReLU()
                self.batch=nn.BatchNorm1d(256)
                self.drop=nn.Dropout(p=0.15) 
                self.fc_x_1 = nn.Linear(256, 128)
                self.fc_x_2 = nn.Linear(128, self.output_horizon)
                self.fc_y_1 = nn.Linear(256, 128)
                self.fc_y_2 = nn.Linear(128, self.output_horizon)
     
        elif self.model_type == 'CNN-LSTM-v1':
            self.init_conv1 = nn.Conv2d(self.in_channel, self.num_channels, kernel_size=self.kernel_size, stride=1, padding = self.padding)  # (100,75)
            self.init_pool2 = nn.MaxPool2d(2, padding = 1) 
            self.init_conv3 = nn.Conv2d(self.num_channels, self.num_channels, kernel_size=self.kernel_size, stride=1, padding = self.padding) # (51,38)
            self.init_pool4 = nn.MaxPool2d(2)
            self.init_conv5 = nn.Conv2d(self.num_channels, self.num_channels, kernel_size=self.kernel_size, stride=1, padding = self.padding) # (25, 19)
            self.init_pool6 = nn.MaxPool2d(2, padding = 1)
            self.init_conv7 = nn.Conv2d(self.num_channels, self.input_horizon, kernel_size=1, stride=1, padding = self.padding) # (25, 19)                    

            self.LSTM_ttlc = nn.GRU((7*35), 512, 1, batch_first = True)
            self.LSTM_fc_x = nn.Linear(512*self.input_horizon, self.output_horizon)
            self.LSTM_fc_y = nn.Linear(512*self.input_horizon, self.output_horizon)

        elif (self.model_type == 'CNN-LSTM-v2'):
            self.init_conv1 = nn.Conv2d(int(self.in_channel/self.input_horizon), self.num_channels, kernel_size=self.kernel_size, stride=1, padding = self.padding)  # (100,75)
            self.init_pool2 = nn.MaxPool2d(2, padding = 1) 
            self.init_conv3 = nn.Conv2d(self.num_channels, self.num_channels, kernel_size=self.kernel_size, stride=1, padding = self.padding) # (51,38)
            self.init_pool4 = nn.MaxPool2d(2)
            self.init_conv5 = nn.Conv2d(self.num_channels, self.num_channels, kernel_size=self.kernel_size, stride=1, padding = self.padding) # (25, 19)
            self.init_pool6 = nn.MaxPool2d(2, padding = 1)        
            self.CNN_fc = nn.Linear(16*5*33, 512)

            self.LSTM_ttlc = nn.GRU(512, 512, 2, batch_first = True)
            self.LSTM_fc = nn.Linear(512, 256)
            self.LSTM_fc_x = nn.Linear(256, self.output_horizon)
            self.LSTM_fc_y = nn.Linear(256, self.output_horizon)
        
        elif self.model_type == 'CNN-LSTM-v3':
            self.init_conv1 = nn.Conv2d(self.in_channel, self.num_channels, kernel_size=self.kernel_size, stride=1, padding = self.padding)  # (100,75)
            self.init_pool2 = nn.MaxPool2d(2, padding = 1) 
            self.init_conv3 = nn.Conv2d(self.num_channels, self.num_channels, kernel_size=self.kernel_size, stride=1, padding = self.padding) # (51,38)
            self.init_pool4 = nn.MaxPool2d(2)
            self.init_conv5 = nn.Conv2d(self.num_channels, self.num_channels, kernel_size=self.kernel_size, stride=1, padding = self.padding) # (25, 19)
            self.init_pool6 = nn.MaxPool2d(2, padding = 1)
        
            self.LSTM_fc = nn.Linear((16*5*33), 512)
            self.LSTM_ttlc = nn.GRU(input_size = 2, hidden_size = 512, num_layers =1, batch_first = True)
            self.LSTM_fc_x_y_1 = nn.Linear(512, 64)
            self.LSTM_fc_x_y_2 = nn.Linear(64, 2)

        elif self.model_type == 'CNN-LSTM-v4':
            self.conv = nn.Sequential(
                
                nn.Conv2d(int(self.in_channel/self.input_horizon), self.num_channels, kernel_size=self.kernel_size, stride=1, padding = self.padding),  # (100,75)
                nn.LeakyReLU(),
                nn.MaxPool2d(2),
                # nn.BatchNorm2d(self.num_channels), 
                
                nn.Conv2d(self.num_channels, self.num_channels, kernel_size=self.kernel_size, stride=1, padding = self.padding), # (51,38)
                nn.LeakyReLU(),
                
                nn.Conv2d(self.num_channels, self.num_channels, kernel_size=self.kernel_size, stride=1, padding = self.padding), # (25, 19)
                nn.LeakyReLU(),
                nn.MaxPool2d(2),
                # nn.BatchNorm2d(self.num_channels), 

                nn.Conv2d(self.num_channels, self.num_channels, kernel_size=self.kernel_size, stride=1, padding = self.padding), # (51,38)
                nn.LeakyReLU(),

                nn.Conv2d(self.num_channels, self.input_horizon, kernel_size=1, stride=1, padding = self.padding),
                nn.LeakyReLU(),
                nn.MaxPool2d(2),
                # nn.BatchNorm2d(self.input_horizon), 
                # nn.Dropout(p=0.15),

                nn.Conv2d(self.input_horizon, self.input_horizon, kernel_size=self.kernel_size, stride=1, padding = self.padding), # (51,38)
                nn.LeakyReLU(),

            ) # (25, 19)                    
            if self.input_horizon == 10:
                self.LSTM_ttlc = nn.GRU((1485), 64, 2, batch_first = True) #46*35
            elif self.input_horizon == 1:
                self.LSTM_ttlc = nn.GRU((165), 64, 2, batch_first = True)
            
            # self.LSTM_batch_norm = nn.BatchNorm1d(512*self.input_horizon)
            self.LSTM_fc_x = nn.Linear(64*self.input_horizon, self.output_horizon)
            self.LSTM_fc_y = nn.Linear(64*self.input_horizon, self.output_horizon)
        
        # elif self.model_type == 'Prob-CNN-LSTM-v4':
        #     self.conv = nn.Sequential(
        #         nn.Conv2d(int(self.in_channel/self.input_horizon), self.num_channels, kernel_size=self.kernel_size, stride=1, padding = self.padding),  # (100,75)
        #         nn.MaxPool2d(2, padding = 1),
        #         nn.BatchNorm2d(self.num_channels), 
        #         nn.Conv2d(self.num_channels, self.num_channels, kernel_size=self.kernel_size, stride=1, padding = self.padding), # (51,38)
        #         nn.MaxPool2d(2),
        #         nn.Conv2d(self.num_channels, self.num_channels, kernel_size=self.kernel_size, stride=1, padding = self.padding), # (25, 19)
        #         nn.MaxPool2d(2, padding = 1),
        #         nn.Conv2d(self.num_channels, self.input_horizon, kernel_size=1, stride=1, padding = self.padding), # (25, 19)                    
        #     )
        #     if self.input_horizon == 10:
        #         self.LSTM_ttlc = nn.GRU((47*36), 32, 2, batch_first = True)
        #     elif self.input_horizon == 1:
        #         self.LSTM_ttlc = nn.GRU((7*35), 32, 2, batch_first = True)
            
            # self.LSTM_batch_norm = nn.BatchNorm1d(512*self.input_horizon)
        elif self.model_type == 'Prob-CNN-LSTM-v4':
            self.conv = nn.Sequential(
                
                nn.Conv2d(int(self.in_channel/self.input_horizon), self.num_channels, kernel_size=self.kernel_size, stride=1, padding = self.padding),  # (100,75)
                nn.LeakyReLU(),
                nn.MaxPool2d(2),
                # nn.BatchNorm2d(self.num_channels), 
                
                nn.Conv2d(self.num_channels, self.num_channels, kernel_size=self.kernel_size, stride=1, padding = self.padding), # (51,38)
                nn.LeakyReLU(),
                
                nn.Conv2d(self.num_channels, self.num_channels, kernel_size=self.kernel_size, stride=1, padding = self.padding), # (25, 19)
                nn.LeakyReLU(),
                nn.MaxPool2d(2),
                # nn.BatchNorm2d(self.num_channels), 

                nn.Conv2d(self.num_channels, self.num_channels, kernel_size=self.kernel_size, stride=1, padding = self.padding), # (51,38)
                nn.LeakyReLU(),

                nn.Conv2d(self.num_channels, self.input_horizon, kernel_size=1, stride=1, padding = self.padding),
                nn.LeakyReLU(),
                nn.MaxPool2d(2),
                # nn.BatchNorm2d(self.input_horizon), 
                # nn.Dropout(p=0.15),

                nn.Conv2d(self.input_horizon, self.input_horizon, kernel_size=self.kernel_size, stride=1, padding = self.padding), # (51,38)
                nn.LeakyReLU(),

            ) # (25, 19)                    
            if self.input_horizon == 10:
                self.LSTM_ttlc = nn.GRU((1485), 64, 2, batch_first = True) #46*35
            elif self.input_horizon == 1:
                self.LSTM_ttlc = nn.GRU((6*35), 64, 2, batch_first = True)
            
            # self.LSTM_batch_norm = nn.BatchNorm1d(512*self.input_horizon)
            self.fc_ttlc_muX =  nn.Linear(64*self.input_horizon ,self.output_horizon)#output_horizon size
            self.fc_ttlc_muY =  nn.Linear(64*self.input_horizon ,self.output_horizon)#output_horizon size
            self.fc_ttlc_sigX = nn.Linear(64*self.input_horizon ,self.output_horizon)#output_horizon size
            self.fc_ttlc_sigY = nn.Linear(64*self.input_horizon ,self.output_horizon)#output_horizon size
            self.fc_ttlc_rho =  nn.Linear(64*self.input_horizon ,self.output_horizon)#output_horizon size
            
        elif self.model_type == 'CNN-LSTM-v5':
            self.init_conv1 = nn.Conv2d(int(self.in_channel/parameters.INPUT_HORIZON), self.num_channels, kernel_size=self.kernel_size, stride=1, padding = self.padding)  # (100,75)
            self.init_pool2 = nn.MaxPool2d(2, padding = 1) 
            self.init_conv3 = nn.Conv2d(self.num_channels, self.num_channels, kernel_size=self.kernel_size, stride=1, padding = self.padding) # (51,38)
            self.init_pool4 = nn.MaxPool2d(2, padding = 1)
            self.init_conv5 = nn.Conv2d(self.num_channels, self.num_channels, kernel_size=self.kernel_size, stride=1, padding = self.padding) # (25, 19)
            self.init_pool6 = nn.MaxPool2d(2, padding = 1)
            self.init_conv6 = nn.Conv2d(self.num_channels, self.num_channels, kernel_size=self.kernel_size, stride=1, padding = self.padding) # (25, 19)
            self.init_pool7 = nn.MaxPool2d(2, padding = 1)
        
            self.LSTM_fc = nn.Linear((16*45*34), 512)
            self.LSTM_ttlc = nn.GRU(input_size = 2, hidden_size = 512, num_layers =1, batch_first = True)
            self.LSTM_fc_x_y_1 = nn.Linear(512, 64)
            self.LSTM_fc_x_y_2 = nn.Linear(64, 2)
        
        elif self.model_type == 'Prob-CNN-LSTM-v6':
            self.init_conv1 = nn.Conv2d(int(3), self.input_horizon, kernel_size=self.kernel_size, stride=1, padding = self.padding)  # (100,75)
            self.init_pool2 = nn.MaxPool2d(2, padding = 1)
            
            self.CNN_batchNorm = nn.BatchNorm2d(self.input_horizon) 
            self.init_conv3 = nn.Conv2d(self.input_horizon, self.input_horizon, kernel_size=self.kernel_size, stride=1, padding = self.padding) # (51,38)
            self.init_pool4 = nn.MaxPool2d(2)
            self.init_conv5 = nn.Conv2d(self.input_horizon, self.input_horizon, kernel_size=self.kernel_size, stride=1, padding = self.padding) # (25, 19)
            self.init_pool6 = nn.MaxPool2d(2, padding = 1)
            self.init_conv7 = nn.Conv2d(self.input_horizon, self.input_horizon, kernel_size=1, stride=1, padding = self.padding) # (25, 19)                    
            
            if self.input_horizon == 10:
                self.mask=torch.zeros((self.input_horizon,self.input_horizon,47,36),device=self.device)
                for i in range (self.input_horizon):
                    self.mask[i,:i+1,:,:] = 1

                self.LSTM_ttlc = nn.GRU((47*36), self.output_horizon, 2, batch_first = True)
            elif self.input_horizon == 1:
                self.mask=torch.zeros((self.input_horizon,self.input_horizon,7,35),device=self.device)
                for i in range (self.input_horizon):
                    self.mask[i,:i+1,:,:] = 1

                self.LSTM_ttlc = nn.GRU((7*35), self.output_horizon, 2, batch_first = True)

            # self.LSTM_batch_norm = nn.BatchNorm1d(512*self.input_horizon)

            self.fc_ttlc_muX =  nn.Linear(self.output_horizon*self.input_horizon ,self.output_horizon)#output_horizon size
            self.fc_ttlc_muY =  nn.Linear(self.output_horizon*self.input_horizon ,self.output_horizon)#output_horizon size
            self.fc_ttlc_sigX = nn.Linear(self.output_horizon*self.input_horizon ,self.output_horizon)#output_horizon size
            self.fc_ttlc_sigY = nn.Linear(self.output_horizon*self.input_horizon ,self.output_horizon)#output_horizon size
            self.fc_ttlc_rho =  nn.Linear(self.output_horizon*self.input_horizon ,self.output_horizon)#output_horizon size

        elif self.model_type == 'CNN-LSTM-v6':
            self.conv = nn.Sequential(
                nn.Conv2d(int(self.in_channel/self.input_horizon), self.input_horizon, kernel_size=self.kernel_size, stride=1, padding = self.padding),                nn.LeakyReLU(),
                nn.MaxPool2d(2, padding = 1),
                nn.Dropout(p=0.15),
                nn.BatchNorm2d(self.input_horizon),

                nn.Conv2d(self.input_horizon, self.input_horizon, kernel_size=self.kernel_size, stride=1, padding = self.padding),
                nn.LeakyReLU(),
                nn.MaxPool2d(2),
                # nn.Dropout(p=0.15),
                # nn.BatchNorm2d(self.input_horizon),
                
                nn.Conv2d(self.input_horizon, self.input_horizon, kernel_size=self.kernel_size, stride=1, padding = self.padding),
                nn.LeakyReLU(),
                nn.MaxPool2d(2, padding = 1),
                # nn.Dropout(p=0.15),
                # nn.BatchNorm2d(self.num_channels),
                
                nn.Conv2d(self.input_horizon, self.input_horizon, kernel_size=1, stride=1, padding = self.padding),
                nn.LeakyReLU(),
                nn.MaxPool2d(2),
                nn.Dropout(p=0.15),
                # nn.BatchNorm2d(self.input_horizon),
            )  
            if self.input_horizon == 10:
                self.mask=torch.zeros((self.input_horizon,self.input_horizon,47,36),device=self.device)
                for i in range (self.input_horizon):
                    self.mask[i,:i+1,:,:] = 1

                self.LSTM_ttlc = nn.GRU((47*36), self.output_horizon, 2, batch_first = True)
            elif self.input_horizon == 1:
                self.mask=torch.zeros((self.input_horizon,self.input_horizon,7,35),device=self.device)
                for i in range (self.input_horizon):
                    self.mask[i,:i+1,:,:] = 1

                self.LSTM_ttlc = nn.GRU((7*35), self.output_horizon, 2, batch_first = True)

            # self.LSTM_batch_norm = nn.BatchNorm1d(512*self.input_horizon)
            self.LSTM_fc_x = nn.Linear(self.output_horizon*self.input_horizon, self.output_horizon)
            self.LSTM_fc_y = nn.Linear(self.output_horizon*self.input_horizon, self.output_horizon)

        elif self.model_type == 'CNN-LSTM-v7':
            self.conv = nn.Sequential(
                nn.Conv2d(int(self.in_channel/self.input_horizon), self.num_channels, kernel_size=self.kernel_size, stride=1, padding = self.padding),
                nn.LeakyReLU(),
                nn.MaxPool2d(2, padding = 1),
                nn.Dropout(p=0.15),
                nn.BatchNorm2d(self.num_channels),

                nn.Conv2d(self.num_channels, self.num_channels, kernel_size=self.kernel_size, stride=1, padding = self.padding),
                nn.LeakyReLU(),
                nn.MaxPool2d(2),
                # nn.Dropout(p=0.15),
                # nn.BatchNorm2d(self.num_channels),
                
                nn.Conv2d(self.num_channels, self.num_channels, kernel_size=self.kernel_size, stride=1, padding = self.padding),
                nn.LeakyReLU(),
                nn.MaxPool2d(2, padding = 1),
                # nn.Dropout(p=0.15),
                # nn.BatchNorm2d(self.num_channels),
                
                nn.Conv2d(self.num_channels, self.input_horizon, kernel_size=1, stride=1, padding = self.padding),
                nn.LeakyReLU(),
                nn.MaxPool2d(2),
                nn.Dropout(p=0.15),
                # nn.BatchNorm2d(self.input_horizon),
            )                    
            if self.input_horizon == 10:
                self.encoder_LSTM_ttlc = nn.GRU((23*18), 16, 2, batch_first = True)
                self.decoder_LSTM_ttlc = nn.GRU((2), 16, 2, batch_first = True)
            
            elif self.input_horizon == 1:
                self.encoder_LSTM_ttlc = nn.GRU((7*35), 16, 2, batch_first = True)
                self.decoder_LSTM_ttlc = nn.GRU((7), 16, 2, batch_first = True)
        
            # self.LSTM_batch_norm = nn.BatchNorm1d(512*self.input_horizon)
            self.LSTM_fc_x_1 = nn.Linear(16*self.output_horizon, 128)
            self.LSTM_fc_y_1 = nn.Linear(16*self.output_horizon, 128)
            self.LSTM_fc_x_2 = nn.Linear(128, self.output_horizon)
            self.LSTM_fc_y_2 = nn.Linear(128, self.output_horizon)

        else:
            print('the model is not implemented yet')
        
        print('model parameters: ', sum(p.numel() for p in self.parameters()))

    def _3D_conv_layer_set(self, in_c, out_c):
            conv_layer = nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size=(3, 3, 3), padding=0),
            nn.LeakyReLU(),
            nn.MaxPool3d((2, 2, 2)),
            )
            return conv_layer
    def _2D_conv_layer_set(self, in_c, out_c):
            conv_layer = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=self.kernel_size, stride=1, padding=self.padding),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(256),
            nn.Dropout(p=0.15)
            )
            return conv_layer
    
    def test(self, x_in):
        if self.model_type == 'CNN-LSTM-v3':
            x = x_in.view(x_in.shape[0], -1, self.image_height, self.image_width)
            x = self.init_conv1(x)
            x = F.relu(self.init_pool2(x)) # F.relu(self.init_pool2(conv1_out))
            x = self.init_conv3(x)
            x = F.relu(self.init_pool4(x)) # F.relu(self.init_pool4(conv3_out))     
            x = self.init_conv5(x) 
            conv_out = F.relu(self.init_pool6(x)) 
            
            # self.lstm(x_input.view(x_input.shape[0], x_input.shape[1], self.input_size))
            x = conv_out.view(conv_out.size(0), (16*5*33))
            encoder_hidden_x = F.relu(self.LSTM_fc(x))

            zeros = torch.zeros([self.batch_size, 2, 1]).to(self.device)
            x, LSTM_hidden = self.LSTM_ttlc(zeros.permute(0,2,1), (encoder_hidden_x.unsqueeze(0)))
            x = F.relu(self.LSTM_fc_x_y_1(x))
            x = self.LSTM_fc_x_y_2(x)
            
            output = x 
            for i in range(1, self.output_horizon):
                x, LSTM_hidden = self.LSTM_ttlc(x, (LSTM_hidden))

                x = F.relu(self.LSTM_fc_x_y_1(x))
                x = self.LSTM_fc_x_y_2(x)
                
                output = torch.cat((output, x), 1) 
            
            out_x = output[:,:,0]
            out_y = output[:,:,1]
            traj_pred = out_x, out_y  
            return {'ttlc_pred':traj_pred, 'features': conv_out}
        
        elif self.model_type == 'CNN-LSTM-v5':
            x_in = torch.transpose(x_in, 1, 2)
            x_grid = torch.stack([make_grid(x_in.unbind(dim=0)[i],padding=3,nrow = 1) for i in range(x_in.shape[0])])

            # x = x_in.view(x_in.shape[0], -1, self.image_height, self.image_width)
            x = self.init_conv1(x_grid)
            x = F.relu(self.init_pool2(x)) # F.relu(self.init_pool2(conv1_out))
            x = self.init_conv3(x)
            x = F.relu(self.init_pool4(x)) # F.relu(self.init_pool4(conv3_out))     
            x = self.init_conv5(x) 
            conv_out = F.relu(self.init_pool6(x)) 
            
            
            # self.lstm(x_input.view(x_input.shape[0], x_input.shape[1], self.input_size))
            # x = conv_out.view(conv_out.size(0), (16*5*33))
            x = conv_out.flatten(1)
            encoder_hidden_x = F.relu(self.LSTM_fc(x))

            zeros = torch.zeros([self.batch_size, 2, 1]).to(self.device)
            x, LSTM_hidden = self.LSTM_ttlc(zeros.permute(0,2,1), (encoder_hidden_x.unsqueeze(0)))
            x = F.relu(self.LSTM_fc_x_y_1(x))
            x = self.LSTM_fc_x_y_2(x)
            
            output = x 
            for i in range(1, self.output_horizon):
                x, LSTM_hidden = self.LSTM_ttlc(x, (LSTM_hidden))

                x = F.relu(self.LSTM_fc_x_y_1(x))
                x = self.LSTM_fc_x_y_2(x)
                
                output = torch.cat((output, x), 1) 
            
            out_x = output[:,:,0]
            out_y = output[:,:,1]
            traj_pred = out_x, out_y  
            return {'ttlc_pred':traj_pred, 'features': conv_out}
        
        elif self.model_type == 'CNN-LSTM-v7': #v3 version with grid horizon input
            x_in = torch.transpose(x_in, 1, 2)
            x_grid = torch.stack([make_grid(x_in.unbind(dim=0)[i],padding=3,nrow = 1) for i in range(x_in.shape[0])])
            conv_out = self.conv(x_grid)
            x = conv_out.flatten(2)
            x,h = self.encoder_LSTM_ttlc(x)
            zeros = torch.zeros([self.batch_size, 2, self.output_horizon]).to(self.device)
            # x = torch.cat((zeros, label_in[:,:, :-1]), -1)
            
            x, LSTM_decoder_hidden = self.decoder_LSTM_ttlc(zeros.permute(0,2,1),h)
            lstm_out = x.flatten(1)
            lstm_out = F.relu(lstm_out)
            x = self.LSTM_fc_x_1(lstm_out)
            y = self.LSTM_fc_y_1(lstm_out)
            x = F.relu(x)
            y = F.relu(y)
            x = self.dropout(x)
            y = self.dropout(y)
            x = self.LSTM_fc_x_2(x)
            y = self.LSTM_fc_y_2(y)
            # output = torch.stack([x,y], axis=2).squeeze(3)
            for i in range(1, self.output_horizon):
                zeros = torch.zeros([self.batch_size, 2, 1]).to(self.device)
                x = torch.cat((zeros.permute(0,2,1), torch.stack([x,y], axis=2)[:,:-1,:]), axis=1)

                x, LSTM_decoder_hidden = self.decoder_LSTM_ttlc(x, (LSTM_decoder_hidden))
                lstm_out = x.flatten(1)
                lstm_out = F.relu(lstm_out)
                x = self.LSTM_fc_x_1(lstm_out)
                y = self.LSTM_fc_y_1(lstm_out)
                x = F.relu(x)
                y = F.relu(y)
                x = self.dropout(x)
                y = self.dropout(y)
                x = self.LSTM_fc_x_2(x)
                y = self.LSTM_fc_y_2(y)
                # output = torch.cat((output, torch.stack([x,y], axis=2).squeeze(3)), 1) 
            traj_pred = x, y
            return {'ttlc_pred':traj_pred, 'features': conv_out}




    def forward(self, x_in, label_in):
        if self.model_type == 'probabilistic':
            x = torch.transpose(x_in, 1, 2).reshape(x_in.shape[0], x_in.shape[1]*x_in.shape[2],
                                                 self.image_height, self.image_width)

            # x = x_in.view(x_in.shape[0], -1, self.image_height, self.image_width)
            x = self.init_conv1(x)
            x = F.relu(self.init_pool2(x)) # F.relu(self.init_pool2(conv1_out))
            x = self.init_conv3(x)
            x = F.relu(self.init_pool4(x)) # F.relu(self.init_pool4(conv3_out))     
            x = self.init_conv5(x) 
            conv_out = F.relu(self.init_pool6(x))             
            
            x = conv_out.view(-1, 16*5*33) # 16*5*33
            x = self.dropout(x)
            x = F.relu(self.fc1_ttlc(x))
            x = self.dropout(x)
            out_muX = self.fc2_ttlc_muX(x)
            out_muY = self.fc2_ttlc_muY(x)
            out_sigX = torch.exp(self.fc2_ttlc_sigX(x))
            out_sigY = torch.exp(self.fc2_ttlc_sigY(x))
            out_rho = torch.tanh(self.fc2_ttlc_rho(x))
            traj_pred = out_muX, out_muY, out_sigX, out_sigY, out_rho

        elif self.model_type == '3DCNN': #v1 version with grid horizon input
            if(self.vector_field_available):
                out_occ_mask = self.conv_occ_mask_layer1(x_in[:,0:1])
                out_vx = self.conv_vx_layer1(x_in[:,1:2])
                out_vy = self.conv_vy_layer1(x_in[:,2:3])
                out_occ_mask = self.conv_occ_mask_layer2(out_occ_mask)
                out_vx = self.conv_vx_layer2(out_vx)
                out_vy = self.conv_vy_layer2(out_vy)
                conv_out = torch.stack([out_occ_mask.flatten(1),out_vx.flatten(1),out_vy.flatten(1)], dim=1)#3*32*1*6*62
                out_fc = conv_out.flatten(1)
                out_fc = self.fc(out_fc)
                out_fc = self.relu(out_fc)
                # out_fc = self.batch(out_fc)
                out_fc = self.drop(out_fc)
                out_fc_x = self.fc_x_1(out_fc)
                out_fc_x = self.relu(out_fc_x)
                out_fc_y = self.fc_y_1(out_fc)
                out_fc_y = self.relu(out_fc_y)
                out_fc_x = self.fc_x_2(out_fc_x)
                out_fc_y = self.fc_y_2(out_fc_y)
            else:
                out_occ_mask = self.conv_occ_mask_layer1(x_in[:,0:1])
                out_occ_mask = self.conv_occ_mask_layer2(out_occ_mask)
                conv_out = out_occ_mask.flatten(1)
                out_fc = self.fc(conv_out)
                out_fc = self.relu(out_fc)
                out_fc = self.batch(out_fc)
                out_fc = self.drop(out_fc)
                out_fc_x = self.fc_x_1(out_fc)
                out_fc_x = self.relu(out_fc_x)
                out_fc_y = self.fc_y_1(out_fc)
                out_fc_y = self.relu(out_fc_y)
                out_fc_x = self.fc_x_2(out_fc_x)
                out_fc_y = self.fc_y_2(out_fc_y)
            traj_pred = out_fc_x, out_fc_y  

        elif self.model_type == 'CNN-Linear':
            # x = x_in.view(x_in.shape[0], -1, self.image_height, self.image_width)
          
            # x = torch.transpose(x_in, 1, 2).reshape(x_in.shape[0], x_in.shape[1]*x_in.shape[2],
            #                                      self.image_height, self.image_width)
          
            x_in = torch.transpose(x_in, 1, 2)
            x_grid = torch.stack([make_grid(x_in.unbind(dim=0)[i],padding=3,nrow = 1) for i in range(x_in.shape[0])])

            x = self.init_conv1(x_grid)
            x = F.relu(self.init_pool2(x)) # F.relu(self.init_pool2(conv1_out))
            x = self.init_conv3(x)
            x = F.relu(self.init_pool4(x)) # F.relu(self.init_pool4(conv3_out))     
            x = self.init_conv5(x) 
            conv_out = F.relu(self.init_pool6(x)) 

            x = conv_out.view(-1, 16*45*34) # 16*5*33
            x = self.dropout(x)
            x = F.relu(self.fc1_ttlc(x))
            x = self.dropout(x)
            out_x = self.fc2_ttlc_x(x)
            out_y = self.fc2_ttlc_y(x)
            traj_pred = out_x, out_y  

        elif self.model_type == 'pretrained_denseNet':
            x_in = torch.transpose(x_in, 1, 2)
            x_grid = torch.stack([make_grid(x_in.unbind(dim=0)[i],padding=3,nrow = 1) for i in range(x_in.shape[0])])

            if self.TrainThePretrained:
                conv_out = self.feat_extract(x_grid)
            else:
                with torch.no_grad():
                    conv_out = self.feat_extract(x_grid)

            x = self.dropout(conv_out)
            out_x = self.fc2_ttlc_x(x)
            out_y = self.fc2_ttlc_y(x)
            traj_pred = out_x, out_y  

        elif self.model_type == 'pretrainedResnetModel':
            x_in = torch.transpose(x_in, 1, 2)
            x_grid = torch.stack([make_grid(x_in.unbind(dim=0)[i],padding=3,nrow = 1) for i in range(x_in.shape[0])])
            if self.TrainThePretrained:
                conv_out = self.feat_extract(x_grid)
            else:
                with torch.no_grad():
                    conv_out = self.feat_extract(x_grid)
            
            x = self.dropout(conv_out)
            out_x = self.fc2_ttlc_x(x)
            out_y = self.fc2_ttlc_y(x)
            traj_pred = out_x, out_y      

        elif (self.model_type == 'Resnet_LSTM'):
            x_in = torch.transpose(x_in, 1, 2).reshape(x_in.shape[0], x_in.shape[1]*x_in.shape[2],
                                                 self.image_height, self.image_width)
            # x = x_in.view(x_in.shape[0], -1, self.image_height, self.image_width)
            hidden = None
            channel_in = int(x_in.shape[1]/self.input_horizon)
            for t in range(self.input_horizon):
                x = x_in[:, t*channel_in : (t+1)*channel_in , :, :]
                if self.TrainThePretrained:
                    conv_out = self.resnet(x)
                else:
                    with torch.no_grad():
                        conv_out = self.resnet(x)
                conv_out = self.dropout(conv_out)
                # x = self.resnet_fc(conv_out)
                out, hidden = self.LSTM_ttlc(conv_out, hidden)

            x = self.LSTM_fc(out)
            out_x = self.LSTM_fc_x(x)
            out_y = self.LSTM_fc_y(x)
        
            traj_pred = out_x, out_y  

        elif self.model_type == 'CNN-LSTM-v1':
            x = torch.transpose(x_in, 1, 2).reshape(x_in.shape[0], x_in.shape[1]*x_in.shape[2],
                                                 self.image_height, self.image_width)
            # x = x_in.view(x_in.shape[0], -1, self.image_height, self.image_width)
            x = self.init_conv1(x)
            x = F.relu(self.init_pool2(x)) # F.relu(self.init_pool2(conv1_out))
            x = self.init_conv3(x)
            x = F.relu(self.init_pool4(x)) # F.relu(self.init_pool4(conv3_out))     
            x = self.init_conv5(x) 
            x = F.relu(self.init_pool6(x)) 
            conv_out = self.init_conv7(x)             
            conv_out = self.dropout(conv_out)

            x = conv_out.flatten(2)
            x = self.LSTM_ttlc(x)[0]
            out_x = self.LSTM_fc_x(x.reshape(self.batch_size, 512*self.input_horizon))
            out_y = self.LSTM_fc_y(x.reshape(self.batch_size, 512*self.input_horizon))
            traj_pred = out_x, out_y  

        elif (self.model_type == 'CNN-LSTM-v2'):
            x_in = torch.transpose(x_in, 1, 2).reshape(x_in.shape[0], x_in.shape[1]*x_in.shape[2],
                                                 self.image_height, self.image_width)
            hidden = None
            channel_in = int(x_in.shape[1]/self.input_horizon)
            for t in range(self.input_horizon):
                x = x_in[:, t*channel_in : (t+1)*channel_in, :, :]
                x = self.init_conv1(x)
                x = F.relu(self.init_pool2(x)) # F.relu(self.init_pool2(conv1_out))
                x = self.init_conv3(x)
                x = F.relu(self.init_pool4(x)) # F.relu(self.init_pool4(conv3_out))     
                x = self.init_conv5(x) 
                conv_out = F.relu(self.init_pool6(x))
                conv_out = self.dropout(conv_out)


                x = conv_out.view(conv_out.size(0), (16*5*33))
                x = self.CNN_fc(x)
                out, hidden = self.LSTM_ttlc(x.squeeze(0), hidden)

            x = self.LSTM_fc(out)
            out_x = self.LSTM_fc_x(x)
            out_y = self.LSTM_fc_y(x)
        
            traj_pred = out_x, out_y  
        
        elif self.model_type == 'CNN-LSTM-v3':
            x = torch.transpose(x_in, 1, 2).reshape(x_in.shape[0], x_in.shape[1]*x_in.shape[2],
                                                 self.image_height, self.image_width)

            # x = x_in.view(x_in.shape[0], -1, self.image_height, self.image_width)
            # fig, ax = plt.subplots(4,1)
            x = self.init_conv1(x)
            x = F.relu(self.init_pool2(x)) # F.relu(self.init_pool2(conv1_out))
            x = self.init_conv3(x)
            x = F.relu(self.init_pool4(x)) # F.relu(self.init_pool4(conv3_out))     
            x = self.init_conv5(x) 
            conv_out = F.relu(self.init_pool6(x)) 
            conv_out = self.dropout(conv_out)

            
            # self.lstm(x_input.view(x_input.shape[0], x_input.shape[1], self.input_size))
            x = conv_out.view(conv_out.size(0), (16*5*33))
            encoder_hidden_x = F.relu(self.LSTM_fc(x))

            zeros = torch.zeros([label_in.shape[0],label_in.shape[1],1]).to(self.device)
            LSTM_input = torch.cat((zeros, label_in[:,:, :-1]), -1)
            x = self.LSTM_ttlc(LSTM_input.permute(0,2,1), (encoder_hidden_x.unsqueeze(0)))[0]
            
            x = F.relu(self.LSTM_fc_x_y_1(x))
            x = self.LSTM_fc_x_y_2(x)

            out_x = x[:,:,0]
            out_y = x[:,:,1]
            traj_pred = out_x, out_y  

        elif self.model_type == 'Prob-CNN-LSTM-v4': #v1 version with grid horizon input
            # x = torch.transpose(x_in, 1, 2).reshape(x_in.shape[0], x_in.shape[1]*x_in.shape[2],
            #                                      self.image_height, self.image_width)
            # x = x_in.view(x_in.shape[0], -1, self.image_height, self.image_width)
            x_in = torch.transpose(x_in, 1, 2)
            x_grid = torch.stack([make_grid(x_in.unbind(dim=0)[i],padding=3,nrow = 1) for i in range(x_in.shape[0])])
            self.grid = x_grid
            self.grid.requires_grad = True
            # x = self.init_conv1(self.grid)
            # x = F.relu(x)
            # x = self.init_pool2(x)
            # # x = self.CNN_batchNorm(x) 
            # x = self.init_conv3(x)
            # x = F.relu(x) # F.relu(self.init_pool2(conv1_out))
            # x = self.init_pool4(x) # F.relu(self.init_pool4(conv3_out))     
            # x = self.init_conv5(x)
            # x = F.relu(x) # F.relu(self.init_pool2(conv1_out))
            # x = self.init_pool6(x)
            # x = self.CNN_batchNorm(x) 
            # conv_out = self.init_conv7(x)             
            # x = F.relu(x) # F.relu(self.init_pool2(conv1_out))
            # conv_out = self.dropout(conv_out)
            conv_out = self.conv(x_grid)
            x = conv_out.flatten(2)
            x = self.LSTM_ttlc(x)[0]
            x = x.flatten(1)
            # x = self.LSTM_batch_norm(x)
            out_muX = self.fc_ttlc_muX(x)
            out_muY = self.fc_ttlc_muY(x)
            out_sigX = torch.exp(self.fc_ttlc_sigX(x))
            out_sigY = torch.exp(self.fc_ttlc_sigY(x))
            out_rho = torch.tanh(self.fc_ttlc_rho(x))
            traj_pred = out_muX, out_muY, out_sigX, out_sigY, out_rho
        
        elif self.model_type == 'CNN-LSTM-v4': #v1 version with grid horizon input
            x_in = torch.transpose(x_in, 1, 2)
            x_grid = torch.stack([make_grid(x_in.unbind(dim=0)[i],padding=3,nrow = 1) for i in range(x_in.shape[0])])

            conv_out = self.conv(x_grid)

            x = conv_out.flatten(2)
            x = self.LSTM_ttlc(x)[0]
            x = x.flatten(1)
            x = self.dropout(x)
            # x = self.LSTM_batch_norm(x)
            out_x = self.LSTM_fc_x(x)
            out_y = self.LSTM_fc_y(x)
            traj_pred = out_x, out_y  

        elif self.model_type == 'CNN-LSTM-v5': #v3 version with grid horizon input
            # x = torch.transpose(x_in, 1, 2).reshape(x_in.shape[0], x_in.shape[1]*x_in.shape[2],
            #                                      self.image_height, self.image_width)
            x_in = torch.transpose(x_in, 1, 2)
            x_grid = torch.stack([make_grid(x_in.unbind(dim=0)[i],padding=3,nrow = 1) for i in range(x_in.shape[0])])

            # x = x_in.view(x_in.shape[0], -1, self.image_height, self.image_width)
            # fig, ax = plt.subplots(4,1)
            x = self.init_conv1(x_grid)
            x = F.relu(self.init_pool2(x)) # F.relu(self.init_pool2(conv1_out))
            x = self.init_conv3(x)
            x = F.relu(self.init_pool4(x)) # F.relu(self.init_pool4(conv3_out))     
            x = self.init_conv5(x) 
            conv_out = F.relu(self.init_pool6(x)) 
            conv_out = self.dropout(conv_out)

            # self.lstm(x_input.view(x_input.shape[0], x_input.shape[1], self.input_size))
            # x = conv_out.view(conv_out.size(0), (16*5*33))
            x = conv_out.flatten(1)
            encoder_hidden_x = F.relu(self.LSTM_fc(x))

            zeros = torch.zeros([label_in.shape[0],label_in.shape[1],1]).to(self.device)
            LSTM_input = torch.cat((zeros, label_in[:,:, :-1]), -1)
            x = self.LSTM_ttlc(LSTM_input.permute(0,2,1), (encoder_hidden_x.unsqueeze(0)))[0]
            
            x = F.relu(self.LSTM_fc_x_y_1(x))
            x = self.LSTM_fc_x_y_2(x)

            out_x = x[:,:,0]
            out_y = x[:,:,1]
            traj_pred = out_x, out_y  

        elif self.model_type == 'Prob-CNN-LSTM-v6': #v4 version with encoder and decoder lstm and grid horizon input
            # x = torch.transpose(x_in, 1, 2).reshape(x_in.shape[0], x_in.shape[1]*x_in.shape[2],
            #                                      self.image_height, self.image_width)
            # x = x_in.view(x_in.shape[0], -1, self.image_height, self.image_width)
            x_in = torch.transpose(x_in, 1, 2)
            x_grid = torch.stack([make_grid(x_in.unbind(dim=0)[i],padding=3,nrow = 1) for i in range(x_in.shape[0])])
            self.grid = x_grid
            self.grid.requires_grad = True
            x = self.init_conv1(self.grid)
            x = F.relu(x)
            x = self.init_pool2(x)
            # x = self.CNN_batchNorm(x) 
            x = self.init_conv3(x)
            x = F.relu(x) # F.relu(self.init_pool2(conv1_out))
            x = self.init_pool4(x) # F.relu(self.init_pool4(conv3_out))     
            x = self.init_conv5(x)
            x = F.relu(x) # F.relu(self.init_pool2(conv1_out))
            x = self.init_pool6(x)
            x = self.CNN_batchNorm(x) 
            conv_out = self.init_conv7(x)
            x = torch.zeros_like(conv_out)
            for i in range (conv_out.shape[1]):
                x[:,i,:,:] = torch.sum(conv_out[:,:i+1, :, :], dim=1)

            x = F.relu(x) # F.relu(self.init_pool2(conv1_out))
            x = self.dropout(x)

            # conv_out = conv_out * self.mask
            x = x.flatten(2)
            x = self.LSTM_ttlc(x)[0]
            
            x = x.flatten(1)
            # x = self.LSTM_batch_norm(x)
            out_muX = self.fc_ttlc_muX(x)
            out_muY = self.fc_ttlc_muY(x)
            out_sigX = torch.exp(self.fc_ttlc_sigX(x))
            out_sigY = torch.exp(self.fc_ttlc_sigY(x))
            out_rho = torch.tanh(self.fc_ttlc_rho(x))
            traj_pred = out_muX, out_muY, out_sigX, out_sigY, out_rho
       
        elif self.model_type == 'CNN-LSTM-v6': #v4 version with encoder and decoder lstm and grid horizon input
            x_in = torch.transpose(x_in, 1, 2)
            x_grid = torch.stack([make_grid(x_in.unbind(dim=0)[i],padding=3,nrow = 1) for i in range(x_in.shape[0])])
            self.grid = x_grid
            self.grid.requires_grad = True
            conv_out = self.conv(self.grid)
            
            x = conv_out.flatten(2)
            x = self.LSTM_ttlc(x)[0]
            
            x = x.flatten(1)
            # x = self.LSTM_batch_norm(x)
            out_x = self.LSTM_fc_x(x)
            out_y = self.LSTM_fc_y(x)
            traj_pred = out_x, out_y 

        elif self.model_type == 'CNN-LSTM-v7': #v3 version with grid horizon input
            x_in = torch.transpose(x_in, 1, 2)
            x_grid = torch.stack([make_grid(x_in.unbind(dim=0)[i],padding=3,nrow = 1) for i in range(x_in.shape[0])])
            conv_out = self.conv(x_grid)
            x = conv_out.flatten(2)
            x,h = self.encoder_LSTM_ttlc(x)
            zeros = torch.zeros([label_in.shape[0],label_in.shape[1],1]).to(self.device)
            x = torch.cat((zeros, label_in[:,:, :-1]), -1)
            x = self.decoder_LSTM_ttlc(x.permute(0,2,1),h)[0]
            lstm_out = x.flatten(1)
            lstm_out = F.relu(lstm_out)
            x = self.LSTM_fc_x_1(lstm_out)
            y = self.LSTM_fc_y_1(lstm_out)
            x = F.relu(x)
            y = F.relu(y)
            x = self.dropout(x)
            y = self.dropout(y)
            x = self.LSTM_fc_x_2(x)
            y = self.LSTM_fc_y_2(y)
        
            traj_pred = x.squeeze(), y.squeeze()  
        
        return {'ttlc_pred':traj_pred, 'features': conv_out}

        # if self.task == params.CLASSIFICATION or self.task == params.DUAL:
        #     lc_pred = self.lc_forward(conv7_out)
        # else:
        #     lc_pred = 0        
        # if self.task == params.REGRESSION or self.task == params.DUAL:
        #     ttlc_pred = self.ttlc_forward(conv7_out)
        # else:
        #     ttlc_pred = 0
        
        # return {'ttlc_pred':traj_pred, 'features': conv_out}

            # self.init_conv1 = nn.Conv2d(self.in_channel, self.num_channels, kernel_size=self.kernel_size, stride=1, padding = self.padding)  # (100,75)
            # self.init_pool2 = nn.MaxPool2d(2, padding = 1) 
            # self.init_conv3 = nn.Conv2d(self.num_channels, self.num_channels, kernel_size=self.kernel_size, stride=1, padding = self.padding) # (51,38)
            # self.init_pool4 = nn.MaxPool2d(2)
            # self.init_conv5 = nn.Conv2d(self.num_channels, self.num_channels, kernel_size=self.kernel_size, stride=1, padding = self.padding) # (25, 19)
            # self.init_pool6 = nn.MaxPool2d(2, padding = 1)
            # # self.init_conv7 = nn.Conv2d(self.num_channels, self.num_channels, kernel_size=self.kernel_size, stride=1, padding = self.padding) # (25, 19)
            # # self.init_pool8 = nn.MaxPool2d(2, padding = 1)
            # self.init_conv7 = nn.Conv2d(self.num_channels, self.input_horizon, kernel_size=1, stride=1, padding = self.padding) # (25, 19)
            # # self.init_pool10 = nn.MaxPool2d(2, padding = 1)
            # self.dropout = nn.Dropout(drop_prob)
            
            # if self.task == params.CLASSIFICATION or self.task == params.DUAL:
            #     self.fc1 = nn.Linear(2*11*13*self.num_channels, 128)
            #     self.fc2 = nn.Linear(128,3)
            
            # if self.task == params.REGRESSION or self.task == params.DUAL:
            #     if (self.probabilistic_model):
            #         # self.fc1_ttlc = nn.Linear(self.batch_size*self.input_horizon*5*49, 512) # 16*5*33
            #         # self.fc2_ttlc_muX = nn.Linear(512,self.output_horizon)#horizen size
            #         # self.fc2_ttlc_muY = nn.Linear(512,self.output_horizon)#horizen size
            #         # self.fc2_ttlc_sigX = nn.Linear(512,self.output_horizon)#horizen size
            #         # self.fc2_ttlc_sigY = nn.Linear(512,self.output_horizon)#horizen size
            #         # self.fc2_ttlc_rho = nn.Linear(512,self.output_horizon)#horizen size
            #     else:
            #         # self.fc1_ttlc = nn.Linear(self.batch_size*self.input_horizon*5*49, 512) # 16*5*33
            #         # self.fc2_ttlc_x = nn.Linear(512,self.output_horizon)#horizen size
            #         # self.fc2_ttlc_y = nn.Linear(512,self.output_horizon)
            
            # self.Fuzzy = FN(args=parameters,device=device,in_features=512, out_features=self.output_horizon*2,
            #                 membership_functions = 6, inference_order=2, )
            # self.batch_size*self.input_horizon*(5*33)

    # def LSTM_forward(self, x_in):
    #     x = x_in.view(x_in.size(0), x_in.size(1), (5*49))
    #     # x = x_in.view(x_in.size(0), x_in.size(1), -1)
    #     # x = x.permute(0, 2, 1)        
    #     # x, (hn, cn) = self.lstm(x, self.hidden)
    #     # y  = self.linear2(x[:, -1, :])
        
    #     # linear_x = self.LSTM_fc(x[:, -1, :])
    #     # out_x = linear_x[:,0]
    #     # out_y = linear_x[:,1]
    #     # return out_x, out_y

    #     # x = x_in.view(-1, 16*5*33)
    #     x = self.LSTM_ttlc(x)[0]
    #     out_x = self.LSTM_fc_x(x.reshape(self.batch_size, 512*self.input_horizon))
    #     out_y = self.LSTM_fc_x(x.reshape(self.batch_size, 512*self.input_horizon))
    #     return out_x, out_y 

    # def conv_forward(self, x_in):
        
    #     x = x_in.view(-1, self.in_channel, self.image_height, self.image_width)
    #     conv1_out = self.init_conv1(x)
    #     conv2_out = F.relu(self.init_pool2(conv1_out)) # F.relu(self.init_pool2(conv1_out))
    #     conv3_out = self.init_conv3(conv2_out)
    #     conv4_out = F.relu(self.init_pool4(conv3_out)) # F.relu(self.init_pool4(conv3_out))     
    #     conv5_out = self.init_conv5(conv4_out) 
    #     conv6_out = F.relu(self.init_pool6(conv5_out))
    #     conv7_out = F.relu(self.init_conv7(conv6_out))  
    #     return (conv1_out, conv2_out, conv3_out, conv4_out, conv5_out, conv6_out, conv7_out)
    
    # def lc_forward(self, conv_out):
    #     x = conv_out.view(-1, 16*5*19) # 16*5*33
    #     x = self.dropout(x)
    #     out = F.relu(self.fc1(x))
    #     out = self.dropout(out)
    #     out = self.fc2(out)
    #     return out
            
    # def ttlc_forward(self, conv_out):
    #     if(self.probabilistic_model):
    #         x = conv_out.view(-1, 16*5*19) # 16*5*33
    #         x = self.dropout(x)
    #         out = F.relu(self.fc1_ttlc(x))
    #         out = self.dropout(out)
    #         out_muX = self.fc2_ttlc_muX(out)
    #         out_muY = self.fc2_ttlc_muY(out)
    #         out_sigX = torch.exp(self.fc2_ttlc_sigX(out))
    #         out_sigY = torch.exp(self.fc2_ttlc_sigY(out))
    #         out_rho = torch.tanh(self.fc2_ttlc_rho(out))
    #         return out_muX, out_muY, out_sigX, out_sigY, out_rho
    #     elif(self.LSTM_model):
    #         # out = F.relu(self.fc1_ttlc(x))
    #         out_x, out_y = self.LSTM_forward(conv_out)
    #         return out_x, out_y
    #     else:
    #         x = conv_out.view(-1, 16*5*19) # 16*5*33
    #         x = self.dropout(x)
    #         out = F.relu(self.fc1_ttlc(x))
    #         out = self.dropout(out)
    #         out_x = self.fc2_ttlc_x(out)
    #         out_y = self.fc2_ttlc_y(out)
    #         return out_x, out_y        

class ATTCNN3(nn.Module):

    def __init__(self, batch_size, device, hyperparams_dict, parameters, drop_prob = 0.5):
        super(ATTCNN3, self).__init__()
        
        self.batch_size = batch_size
        self.device = device
        # Hyperparams:
        self.num_channels = hyperparams_dict['channel number']
        self.kernel_size = hyperparams_dict['kernel size']
        self.single_merged_ch = hyperparams_dict['merge channels']
        self.task = hyperparams_dict['task']
        self.padding = int((self.kernel_size -1)/2) 

        self.in_seq_len = parameters.IN_SEQ_LEN
        self.image_height = parameters.IMAGE_HEIGHT
        self.image_width = parameters.IMAGE_WIDTH 
        # Initial Convs
        if self.single_merged_ch:
            self.in_channel = self.in_seq_len


        self.init_conv1 = nn.Conv2d(self.in_channel,self.num_channels, kernel_size=self.kernel_size, stride=1, padding = self.padding)  # (100,75)
        self.init_pool2 = nn.MaxPool2d(2, padding = 1) 
        self.init_conv3 = nn.Conv2d(self.num_channels, self.num_channels, kernel_size=self.kernel_size, stride=1, padding = self.padding) # (51,38)
        self.init_pool4 = nn.MaxPool2d(2)
        self.init_conv5 = nn.Conv2d(self.num_channels, self.num_channels, kernel_size=self.kernel_size, stride=1, padding = self.padding) # (25, 19)
        self.init_pool6 = nn.MaxPool2d(2, padding = 1)
        self.dropout = nn.Dropout(drop_prob)
        self.fc1 = nn.Linear(2*11*13*self.num_channels, 4)

        if self.task == params.CLASSIFICATION or self.task == params.DUAL:
            self.fc3 = nn.Linear(12*26*self.num_channels,128)# 6*13
            self.fc4 = nn.Linear(128, 3)

        if self.task == params.REGRESSION or self.task == params.DUAL:
            self.fc1_ttlc = nn.Linear(12*26*self.num_channels, 512)
            self.fc2_ttlc = nn.Linear(512,1)

    def conv_forward(self, x_in):
        
        x_in = x_in[0]
        if self.single_merged_ch:
            x_in = torch.mean(x_in, 2, True)
        x = x_in.view(-1, self.in_channel, self.image_height, self.image_width)
        conv1_out = self.init_conv1(x)
        conv2_out = F.relu(self.init_pool2(conv1_out))
        conv3_out = self.init_conv3(conv2_out)
        conv4_out = F.relu(self.init_pool4(conv3_out))     
        conv5_out = self.init_conv5(conv4_out) 
        conv6_out = F.relu(self.init_pool6(conv5_out)) 
        
        return conv6_out
    
    def attention_coef_forward(self, conv_out):
        x = conv_out.view(-1, 2*11*13*self.num_channels)
        out = F.softmax(self.fc1(x), dim = -1)
        return out

    def lc_forward(self, attended_features):        
        x = attended_features.view(-1, 26*12*self.num_channels)
        x = self.dropout(x)
        out = F.relu(self.fc3(x))
        out = self.dropout(out)
        out = self.fc4(out)
        return out

    def ttlc_forward(self, attended_features):
        
        x = attended_features.view(-1, 26*12*self.num_channels)
        x = self.dropout(x)
        out = F.relu(self.fc1_ttlc(x))
        out = self.dropout(out)
        out = F.relu(self.fc2_ttlc(out))
        return out

    
    def forward(self,x_in, seq_itr = 0):
        conv6_out= self.conv_forward(x_in)
        
        attention_coef = self.attention_coef_forward(conv6_out)
        front_right = conv6_out[:,:,:6, :13]
        front_left = conv6_out[:,:,5:,:13]
        back_right = conv6_out[:,:,:6,13:]
        back_left = conv6_out[:,:,5:,13:]
        
        front_right_coef = attention_coef[:,0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(*front_right.size())
        front_left_coef = attention_coef[:,1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(*front_right.size())
        back_right_coef = attention_coef[:,2].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(*front_right.size())
        back_left_coef = attention_coef[:,3].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(*front_right.size())
        if seq_itr<10:
            front_right =front_right * front_right_coef
            front_left =front_left * front_left_coef
            back_right =back_right * back_right_coef
            back_left =back_left * back_left_coef

        front = torch.cat((front_right, front_left), dim = 2)
        back = torch.cat((back_right, back_left), dim = 2)
        attended_features = torch.cat((front, back), dim = 3)
        
        if self.task == params.CLASSIFICATION or self.task == params.DUAL:
            lc_pred = self.lc_forward(attended_features)
        else:
            lc_pred = 0
        
        if self.task == params.REGRESSION or self.task == params.DUAL:
            ttlc_pred = self.ttlc_forward(attended_features)
        else:
            ttlc_pred = 0
        return {'lc_pred':lc_pred, 'ttlc_pred':ttlc_pred, 'features': conv6_out, 'attention': attention_coef}
        

