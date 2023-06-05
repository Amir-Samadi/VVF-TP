import os
import matplotlib.pyplot as plt
import logging
from time import time
import numpy as np 
import pickle
import shutil 
import random
from data_loader import CustomImageDataset
import torch
import torch.utils.data as utils_data
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import  roc_curve, auc, roc_auc_score
import matplotlib
import matplotlib.pyplot as plt
import csv
import Dataset 
import models 
import mat73
import params
import models_dict as m
import utils
import tqdm
from copy import deepcopy
import torch.multiprocessing
from torchmetrics import MeanSquaredLogError
torch.multiprocessing.set_sharing_strategy('file_system')
from utils import NLL_loss

import matplotlib.colors as mcolors


def get_mean_and_std(dataloader):
    max_vx = -10000
    min_vx = 10000
    max_vy = -10000
    min_vy = 10000
    max_label_x = -10000
    min_label_x = 10000
    max_label_y = -10000
    min_label_y = 10000
    incremental_mu=0
    incremental_std=0
    incremental_var=0
    incremental_mu_label=0
    incremental_std_label=0
    incremental_var_label=0

    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    label_channels_sum, label_channels_squared_sum, label_num_batches = 0, 0, 0
    for batch_idx, data in enumerate(dataloader):
        num_batches += 1
        
        data_occ = data['occ']
        data_mask = data['mask']
        data_v_x  = data['v_x']
        data_v_y  = data['v_y']
        label_x_fut = data['x_fut']
        label_y_fut = data['y_fut']
        ch_data = torch.cat((data_occ, data_mask, data_v_x, data_v_y), 1)
        ch_label = torch.cat((label_x_fut, label_y_fut), 1)
        # Mean over batch, height and width, but not over the channels
        channels_sum += torch.mean(ch_data, dim=[0,2,3,4])
        channels_squared_sum += torch.mean(ch_data**2, dim=[0,2,3,4])
        label_channels_sum += torch.mean(ch_label, dim=[0,2])
        label_channels_squared_sum += torch.mean(ch_label**2, dim=[0,2])

        # incremental_mu_0 = deepcopy(incremental_mu)
        # incremental_mu = (torch.mean(ch_data, dim=[0,2,3,4]) + incremental_mu_0*(num_batches-1))/num_batches
        # incremental_var = incremental_var + (torch.mean(ch_data, dim=[0,2,3,4]) - incremental_mu_0) * (torch.mean(ch_data, dim=[0,2,3,4]) - incremental_mu)  
        # incremental_std = (incremental_var/num_batches)**0.5

        
        # incremental_mu_label_0 = deepcopy(incremental_mu_label)
        # incremental_mu_label = (torch.mean(ch_label, dim=[0,2]) + incremental_mu_label_0*(num_batches-1))/num_batches
        # incremental_var_label = incremental_var_label + (torch.mean(ch_label, dim=[0,2]) - incremental_mu_label_0) * (torch.mean(ch_label, dim=[0,2]) - incremental_mu_label)  
        # incremental_std_label = (incremental_var_label/num_batches)**0.5


        max_vx = data_v_x.max() if (data_v_x.max() > max_vx) else max_vx 
        min_vx = data_v_x.min() if (data_v_x.min() < min_vx) else min_vx 
        max_vy = data_v_y.max() if (data_v_y.max() > max_vy) else max_vy 
        min_vy = data_v_y.min() if (data_v_y.min() < min_vy) else min_vy 

        max_label_x = label_x_fut.max() if (label_x_fut.max() > max_label_x) else max_label_x 
        min_label_x = label_x_fut.min() if (label_x_fut.min() < min_label_x) else min_label_x 
        max_label_y = label_y_fut.max() if (label_y_fut.max() > max_label_y) else max_label_y 
        min_label_y = label_y_fut.min() if (label_y_fut.min() < min_label_y) else min_label_y 

    mean = channels_sum / num_batches
    label_mean = label_channels_sum / num_batches

    # std = sqrt(E[X^2] - (E[X])^2)
    std = ((channels_squared_sum / num_batches) - (mean ** 2)) ** 0.5
    label_std = ((label_channels_squared_sum / num_batches) - (label_mean ** 2)) ** 0.5

    return mean.cpu().numpy(), std.cpu().numpy(), max_vx.cpu().numpy(), min_vx.cpu().numpy(), max_vy.cpu().numpy(), min_vy.cpu().numpy(),\
        max_label_x.cpu().numpy(), min_label_x.cpu().numpy(), max_label_y.cpu().numpy(), min_label_y.cpu().numpy(), label_mean.cpu().numpy(),\
              label_std.cpu().numpy()

class NRMSE(nn.Module): ## implementation of NRMSE with standard deviation
    def __init__(self):
        super().__init__()
    def forward(true, pred):
        squared_error = torch.square((true - pred))
        sum_squared_error = torch.sum(squared_error)
        rmse = torch.sqrt(sum_squared_error / true.size)
        nrmse_loss = rmse/torch.std(pred)
        return nrmse_loss

class RMSLELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self, pred, actual):
        return torch.sqrt(self.mse(torch.log(pred + 1), torch.log(actual + 1)))

def train_model_dict(model_dict, p):
    # Set Random Seeds:
    if torch.cuda.is_available() and p.CUDA:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            torch.cuda.manual_seed_all(0)
    else:
        device = torch.device("cpu")
            
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True
    np.random.seed(1)
    random.seed(1)

    # val_data_loader = torch.utils.data.DataLoader(val_dataset,batch_size=1, shuffle=True,
    #                                                num_workers=1, drop_last=True)
    
    # tr_dataset = Dataset.LCDataset(p.TRAIN_DATASET_DIR, p.TR_DATA_FILES, data_type = model_dict['data type'], state_type = model_dict['state type'], keep_plot_info= False)
    # #print(tr_dataset.states_max-tr_dataset.states_min)
    # #assert np.all((tr_dataset.states_max-tr_dataset.states_min)>0)
    # #exit()
    # val_dataset = Dataset.LCDataset(p.TRAIN_DATASET_DIR, p.VAL_DATA_FILES,  data_type = model_dict['data type'], state_type = model_dict['state type'], keep_plot_info= False, states_min = tr_dataset.states_min, states_max = tr_dataset.states_max)
    # te_dataset = Dataset.LCDataset(p.TEST_DATASET_DIR, p.TE_DATA_FILES,  data_type = model_dict['data type'], state_type = model_dict['state type'], keep_plot_info= True, states_min = tr_dataset.states_min, states_max = tr_dataset.states_max)
    # #print(tr_dataset.__len__())
    # #print(val_dataset.__len__())
    # #print(te_dataset.__len__())
    # #exit()
    # # Train/Evaluate:

    
    models_results = {'tag':[], 'validation loss':[]}
    print ('start loading dataset')
    dataFileName = 'AVs_NoVF'
    preprocessData = False
    raw_data = None
    if(preprocessData):
        raw_data = mat73.loadmat(dataFileName + '.mat')['AVs']
    
    # raw_data = mat73.loadmat('AVs_4.mat')['AVs']
    # raw_data = mat73.loadmat('AVs_short.mat')['AVs']
    
    for model_type in   [ 
                        # 'Resnet_LSTM', 
                        # 'pretrainedResnetModel',
                        # 'pretrained_denseNet',
                        # 'probabilistic' ,
                        #   'CNN-LSTM-v3',
                        #   'CNN-LSTM-v1',
                          'CNN-LSTM-v4',
                        # 'Prob-CNN-LSTM-v4'
                        #   'CNN-LSTM-v5',
                        #  'CNN-LSTM-v2',
                        # 'CNN-Linear'
                        # 'CNN-LSTM-v6'
                        # 'Prob-CNN-LSTM-v6'
                        # '3DCNN'
                        # 'CNN-LSTM-v7'
                        ]:
        model_dict['hyperparams']['model type'] = model_type
        if model_type in ['probabilistic', 'Prob-CNN-LSTM-v4', 'Prob-CNN-LSTM-v6']:
            model_dict['ttlc loss function'] = NLL_loss
        else:
            model_dict['ttlc loss function'] = torch.nn.HuberLoss()
            # model_dict['ttlc loss function'] = torch.nn.MSELoss()
            # model_dict['ttlc loss function'] = RMSLELoss()
        
        # for vector_field in [False, True]:
        for vector_field in [False]:
            p.vector_field_available = vector_field
        
            # for input_horizon in [int(10)]:
            for input_horizon in [int(10)]:
                p.INPUT_HORIZON = input_horizon
                print ('start Instantiating Dataset')
                # Instantiate Dataset:
                
                DataSavePath = os.path.join('data',dataFileName, 'input_horizon_'+str(input_horizon)) 
                # DataSavePath = os.path.join('/media/samadi_a/T7/reza/data3', dataFileName, 'input_horizon_'+str(input_horizon)) 
                dataset = CustomImageDataset(raw_data = raw_data ,
                                            label_horizen = p.SEQ_LEN-p.IN_SEQ_LEN+2,
                                            input_horizen = p.INPUT_HORIZON,
                                            min_vehicle_frame_exist=p.INPUT_HORIZON,
                                            p=p, DataSavePath=DataSavePath)
                if not os.path.exists(DataSavePath):
                    os.makedirs(DataSavePath)
                if preprocessData:
                    dataset.prepare()
                
                print ('Dataset Len:', len(dataset))
                train_length=int(0.7*len(dataset))
                test_length=len(dataset)-train_length
                
                # the data set generators should go next to the raw data definition,
                # then it should save processed dataset and loads if it is already done,
                # and pass them to the dataloaders 
                tr_dataset = torch.utils.data.Subset(dataset, range(train_length))
                val_dataset = torch.utils.data.Subset(dataset, range(train_length, train_length+test_length ))


                tr_loader = utils_data.DataLoader(dataset = tr_dataset, shuffle = True, batch_size = p.BATCH_SIZE,
                                                drop_last= True, pin_memory= True, num_workers= 20)
                val_loader = utils_data.DataLoader(dataset = val_dataset, shuffle = False, batch_size = p.BATCH_SIZE,
                                                    drop_last= True, pin_memory= True, num_workers= 20)

                
                # #####################################################
                # mean, std, max_vx, min_vx, max_vy, min_vy,max_label_x, min_label_x, max_label_y, min_label_y, label_mean, label_std =\
                #       get_mean_and_std(tr_loader)
                # print(  "mean: ", mean
                #       , "std: ", std
                #       , "max_vx: " , max_vx
                #       , "min_vx: " , min_vx
                #       , "max_vy: " , max_vy
                #       , "min_vy: " , min_vy
                #       , "max_label_x: " , max_label_x
                #       , "min_label_x: " , min_label_x
                #       , "max_label_y: " , max_label_y
                #       , "min_label_y: " , min_label_y
                #       , "label_mean: "  , label_mean
                #       , "label_std: " , label_std )
                # exit()


                # mean:  [ 4.9260162e-02  6.9080950e-03 -1.3276902e+00  1.2969130e-05]
                # std:  [0.21641071 0.08282737 5.9344087  0.00651047]
                # max_vx:  0.0        min_vx:  -63.77         max_vy:  0.314      min_vy:  -0.358 
                # max_label_x:  -3.85 min_label_x:  -10.68    max_label_y:  0.33  min_label_y:  -0.39 
                # label_mean:  [-5.6779733e+00 -1.0209690e-03] 
                # label_std:  [1.0219698  0.03529745]

                # #####################################################
                
                if model_type in ['Resnet_LSTM', 'pretrainedResnetModel', 'pretrained_denseNet']:
                    for pretrain in [False]: #[False, True]
                        p.TrainThePretrained = pretrain
                        tag = model_type + '_TrainPretrained_' + str(pretrain) + '_VectorField_' + str(p.vector_field_available)\
                        +'_inputHorizon_' + str(p.INPUT_HORIZON)
                        print('#########################################################################')
                        print('TRAINING MODEL TAG {}'.format(tag))
                        print('#########################################################################')
                        model_dict['tag'] = tag

                        # create result path
                        p.MODELS_DIR = 'models/'
                        p.RESULTS_DIR = os.path.join('results/' , tag)

                        p.TABLES_DIR = os.path.join(p.RESULTS_DIR, 'tables')
                        p.FIGS_DIR = os.path.join(p.RESULTS_DIR , 'figures')
                        p.VIS_DIR = os.path.join(p.RESULTS_DIR , 'vis_data')

                        p.sample_image_DIR = os.path.join(p.RESULTS_DIR , 'train/')
                        p.test_image_DIR = os.path.join(p.RESULTS_DIR , 'test/')
                        p.last_model_DIR = os.path.join(p.RESULTS_DIR , 'model/')

                        if not os.path.exists(p.MODELS_DIR):
                            os.mkdir(p.MODELS_DIR)
                        if not os.path.exists(p.RESULTS_DIR):
                            os.mkdir(p.RESULTS_DIR)
                        if not os.path.exists(p.TABLES_DIR):
                            os.mkdir(p.TABLES_DIR)
                        if not os.path.exists(p.FIGS_DIR):
                            os.mkdir(p.FIGS_DIR)
                        if not os.path.exists(p.VIS_DIR):
                            os.mkdir(p.VIS_DIR)
                        if not os.path.exists(p.sample_image_DIR):
                            os.mkdir(p.sample_image_DIR)
                        else:
                            shutil.rmtree(p.sample_image_DIR)
                            os.mkdir(p.sample_image_DIR)
                        if not os.path.exists(p.test_image_DIR):
                            os.mkdir(p.test_image_DIR)
                        else:
                            shutil.rmtree(p.test_image_DIR)
                            os.mkdir(p.test_image_DIR)
                        if not os.path.exists(p.last_model_DIR):
                            os.mkdir(p.last_model_DIR)     
                

                        # Instantiate Model:
                        model = model_dict['ref'](p.BATCH_SIZE, device, model_dict['hyperparams'], p, drop_prob=0.2).to(device)
                        my_list = [ 'LSTM_ttlc', 'encoder_LSTM_ttlc', 'decoder_LSTM_ttlc', 'LSTM_fc_x', 'LSTM_fc_y']
                        params = list(filter(lambda kv: kv[0].split(".",1)[0] in my_list, model.named_parameters()))
                        base_params = list(filter(lambda kv: kv[0].split(".",1)[0] not in my_list, model.named_parameters()))
                        
                        optimizer = model_dict['optimizer'](
                                                            # [{'params': [model.get_parameter(name[0]) for name in base_params], 'lr': p.LR},
                                                            #     {'params': [model.get_parameter(name[0]) for name in params], 'lr': p.LR/10}],
                                                            model.parameters()
                                                            , lr = p.LR, weight_decay= 1e-4)
                        # optimizer = model_dict['optimizer'](params = model.parameters(), lr = p.LR)
                        lc_loss_func = model_dict['lc loss function']()
                        ttlc_loss_func = model_dict['ttlc loss function']
                        task = model_dict['hyperparams']['task']
                        curriculum_loss = model_dict['hyperparams']['curriculum loss']
                        curriculum_seq = model_dict['hyperparams']['curriculum seq']
                        curriculum_virtual = model_dict['hyperparams']['curriculum virtual']
                        curriculum= {'seq':curriculum_seq, 'loss':curriculum_loss, 'virtual': curriculum_virtual}
                
                        best_model_path = p.MODELS_DIR + '_' + model_dict['tag'] + '.pt'
                        
                        val_result_dic = utils.train_top_func(p, model, optimizer,\
                             lc_loss_func, ttlc_loss_func, task, curriculum,
                         tr_loader, val_loader,device, model_tag = model_dict['tag'])    
                        
                        with open(os.path.join(p.RESULTS_DIR, 'Validation_Loss_History.txt'), 'w') as file:
                            for row in val_result_dic['Validation Loss History']:
                                # s = " ".join(map(str, row))
                                file.write(str(row)+'\n')
                        # Save results:
                        log_file_dir = p.TABLES_DIR + p.SELECTED_DATASET + '_' + model_dict['name'] + '.csv'  
                        log_dict = model_dict['hyperparams'].copy()
                        log_dict['state type'] = model_dict['state type']
                        
                        log_dict.update(val_result_dic)
                        # log_dict.update(te_result_dic)
                        log_columns = [key for key in log_dict]
                        log_columns = ', '.join(log_columns) + '\n'
                        result_line = [str(log_dict[key]) for key in log_dict]
                        result_line = ', '.join(result_line) + '\n'
                        if os.path.exists(log_file_dir) == False:
                            result_line = log_columns + result_line
                        with open(log_file_dir, 'a') as f:
                            f.write(result_line)
   
                else:
                        tag = model_type + '_VectorField_' + str(p.vector_field_available)\
                        +'_inputHorizon_' + str(p.INPUT_HORIZON)
                        model_dict['tag'] = tag
                        print('#########################################################################')
                        print('TRAINING MODEL TAG {}'.format(tag))
                        print('#########################################################################')

                        # create result path
                        p.MODELS_DIR = 'models/'
                        p.RESULTS_DIR = os.path.join('results/' , tag)

                        p.TABLES_DIR = os.path.join(p.RESULTS_DIR, 'tables')
                        p.FIGS_DIR = os.path.join(p.RESULTS_DIR , 'figures')
                        p.VIS_DIR = os.path.join(p.RESULTS_DIR , 'vis_data')

                        p.sample_image_DIR = os.path.join(p.RESULTS_DIR , 'train/')
                        p.test_image_DIR = os.path.join(p.RESULTS_DIR , 'test/')
                        p.last_model_DIR = os.path.join(p.RESULTS_DIR , 'model/')

                        if not os.path.exists(p.MODELS_DIR):
                            os.mkdir(p.MODELS_DIR)
                        if not os.path.exists(p.RESULTS_DIR):
                            os.mkdir(p.RESULTS_DIR)
                        if not os.path.exists(p.TABLES_DIR):
                            os.mkdir(p.TABLES_DIR)
                        if not os.path.exists(p.FIGS_DIR):
                            os.mkdir(p.FIGS_DIR)
                        if not os.path.exists(p.VIS_DIR):
                            os.mkdir(p.VIS_DIR)
                        if not os.path.exists(p.sample_image_DIR):
                            os.mkdir(p.sample_image_DIR)
                        else:
                            shutil.rmtree(p.sample_image_DIR)
                            os.mkdir(p.sample_image_DIR)
                        if not os.path.exists(p.test_image_DIR):
                            os.mkdir(p.test_image_DIR)
                        else:
                            shutil.rmtree(p.test_image_DIR)
                            os.mkdir(p.test_image_DIR)
                        if not os.path.exists(p.last_model_DIR):
                            os.mkdir(p.last_model_DIR)     
                

                        # Instantiate Model:
                        model = model_dict['ref'](p.BATCH_SIZE, device, model_dict['hyperparams'], p).to(device)
                        my_list = [ 'LSTM_ttlc', 'encoder_LSTM_ttlc', 'decoder_LSTM_ttlc']
                        params = list(filter(lambda kv: kv[0].split(".",1)[0] in my_list, model.named_parameters()))
                        base_params = list(filter(lambda kv: kv[0].split(".",1)[0] not in my_list, model.named_parameters()))
                        
                        optimizer = model_dict['optimizer']([
                                                                {'params': [model.get_parameter(name[0]) for name in base_params], 'lr': p.LR},
                                                                {'params': [model.get_parameter(name[0]) for name in params], 'lr': p.LR/4}
                                                            ], lr = p.LR, weight_decay= 1e-4)
                        lc_loss_func = model_dict['lc loss function']()
                        ttlc_loss_func = model_dict['ttlc loss function']
                        task = model_dict['hyperparams']['task']
                        curriculum_loss = model_dict['hyperparams']['curriculum loss']
                        curriculum_seq = model_dict['hyperparams']['curriculum seq']
                        curriculum_virtual = model_dict['hyperparams']['curriculum virtual']
                        curriculum= {'seq':curriculum_seq, 'loss':curriculum_loss, 'virtual': curriculum_virtual}
                
                        best_model_path = p.MODELS_DIR + '_' + model_dict['tag'] + '.pt'
                        val_result_dic = utils.train_top_func(p, model, optimizer, lc_loss_func, ttlc_loss_func, task,\
                             curriculum, tr_loader, val_loader, device, model_tag = model_dict['tag'])    
                        
                        # with open(os.path.join(p.RESULTS_DIR, 'Validation_Loss_History.txt'), 'w') as file:
                        #     for row in val_result_dic['Validation Loss History']:
                        #         # s = " ".join(map(str, row))
                        #         file.write(str(row)+'\n')
                        # Save results:
                        log_file_dir = p.TABLES_DIR + p.SELECTED_DATASET + '_' + model_dict['name'] + '.csv'  
                        log_dict = model_dict['hyperparams'].copy()
                        log_dict['state type'] = model_dict['state type']
                        
                        log_dict.update(val_result_dic)
                        # log_dict.update(te_result_dic)
                        log_columns = [key for key in log_dict]
                        log_columns = ', '.join(log_columns) + '\n'
                        result_line = [str(log_dict[key]) for key in log_dict]
                        result_line = ', '.join(result_line) + '\n'
                        if os.path.exists(log_file_dir) == False:
                            result_line = log_columns + result_line
                        with open(log_file_dir, 'a') as f:
                            f.write(result_line)
                
                # del dataset.data
                
                del dataset.data, model
                models_results['tag'].append(tag) 
                models_results['validation loss'].append(val_result_dic['Validation Loss History'])

                plt.plot(val_result_dic['Validation Loss History'], label=tag)
    plt.title("Validation Loss History")
    plt.legend()
    plt.savefig("Validation Loss History.png")

    with open('Models_results.csv', 'w') as f:
        w = csv.DictWriter(f, models_results.keys())
        w.writeheader()
        for idx in range(len(models_results['tag'])): 
            w.writerow({'tag':models_results['tag'][idx],
                        'validation loss':models_results['validation loss'][idx]})








    ####################################################################
    ####################################################################
    ################################backup##############################
    # best_model_path = p.MODELS_DIR + p.SELECTED_DATASET + '_' + model_dict['tag'] + '.pt'
    # # model.load_state_dict(torch.load(best_model_path))
    # val_result_dic = utils.train_top_func(p, model, optimizer, lc_loss_func, ttlc_loss_func, task, curriculum, tr_dataset, val_dataset,device, model_tag = model_dict['tag'])    
    # # te_result_dic = utils.eval_top_func(p, model, lc_loss_func, ttlc_loss_func, task, val_dataset, device, model_tag = model_dict['tag'])
    
    # # Save results:
    # log_file_dir = p.TABLES_DIR + p.SELECTED_DATASET + '_' + model_dict['name'] + '.csv'  
    # log_dict = model_dict['hyperparams'].copy()
    # log_dict['state type'] = model_dict['state type']
    # log_dict.update(val_result_dic)
    # # log_dict.update(te_result_dic)
    # log_columns = [key for key in log_dict]
    # log_columns = ', '.join(log_columns) + '\n'
    # result_line = [str(log_dict[key]) for key in log_dict]
    # result_line = ', '.join(result_line) + '\n'
    # if os.path.exists(log_file_dir) == False:
    #     result_line = log_columns + result_line
    # with open(log_file_dir, 'a') as f:
    #     f.write(result_line)
    ################################backup##############################
    ####################################################################
    ####################################################################

if __name__ == '__main__':
    
    
    p = params.Parameters(SELECTED_MODEL = 'VCNN', SELECTED_DATASET = 'HIGHD', UNBALANCED = False, ABLATION = False)

    #1
    model_dict = m.MODELS[p.SELECTED_MODEL]

    model_dict['hyperparams']['task'] = params.REGRESSION
    model_dict['hyperparams']['curriculum loss'] = False
    model_dict['hyperparams']['curriculum seq'] = False
    model_dict['hyperparams']['curriculum virtual'] = False

    model_dict['tag'] = utils.update_tag(model_dict)

    train_model_dict(model_dict, p)
    
    exit()

    p = params.Parameters(SELECTED_MODEL = 'VLSTM', SELECTED_DATASET = 'HIGHD', UNBALANCED = False, ABLATION = False)

    #1
    model_dict = m.MODELS[p.SELECTED_MODEL]

    model_dict['hyperparams']['task'] = params.CLASSIFICATION
    model_dict['hyperparams']['curriculum loss'] = False
    model_dict['hyperparams']['curriculum seq'] = False
    model_dict['hyperparams']['curriculum virtual'] = False

    model_dict['tag'] = utils.update_tag(model_dict)

    train_model_dict(model_dict, p)

    p = params.Parameters(SELECTED_MODEL = 'REGIONATTCNN3', SELECTED_DATASET = 'HIGHD', UNBALANCED = False, ABLATION = False)
    #1
    model_dict = m.MODELS[p.SELECTED_MODEL]

    model_dict['hyperparams']['task'] = params.DUAL
    model_dict['hyperparams']['curriculum loss'] = True
    model_dict['hyperparams']['curriculum seq'] = True
    model_dict['hyperparams']['curriculum virtual'] = False

    model_dict['tag'] = utils.update_tag(model_dict)

    train_model_dict(model_dict, p)
   
    
    model_dict['hyperparams']['task'] = params.REGRESSION
    model_dict['hyperparams']['curriculum loss'] = False
    model_dict['hyperparams']['curriculum seq'] = True
    model_dict['hyperparams']['curriculum virtual'] = False

    model_dict['tag'] = utils.update_tag(model_dict)

    train_model_dict(model_dict, p)

    model_dict['hyperparams']['task'] = params.CLASSIFICATION
    model_dict['hyperparams']['curriculum loss'] = True
    model_dict['hyperparams']['curriculum seq'] = True
    model_dict['hyperparams']['curriculum virtual'] = False

    model_dict['tag'] = utils.update_tag(model_dict)

    train_model_dict(model_dict, p)

    p = params.Parameters(SELECTED_MODEL = 'VLSTM', SELECTED_DATASET = 'HIGHD', UNBALANCED = False, ABLATION = False)

    #1
    model_dict = m.MODELS[p.SELECTED_MODEL]

    model_dict['hyperparams']['task'] = params.REGRESSION
    model_dict['hyperparams']['curriculum loss'] = False
    model_dict['hyperparams']['curriculum seq'] = False
    model_dict['hyperparams']['curriculum virtual'] = False

    model_dict['tag'] = utils.update_tag(model_dict)

    train_model_dict(model_dict, p)

    p = params.Parameters(SELECTED_MODEL = 'VLSTM', SELECTED_DATASET = 'HIGHD', UNBALANCED = False, ABLATION = False)

    #1
    model_dict = m.MODELS[p.SELECTED_MODEL]

    model_dict['hyperparams']['task'] = params.CLASSIFICATION
    model_dict['hyperparams']['curriculum loss'] = False
    model_dict['hyperparams']['curriculum seq'] = False
    model_dict['hyperparams']['curriculum virtual'] = False

    model_dict['tag'] = utils.update_tag(model_dict)

    train_model_dict(model_dict, p)

    p = params.Parameters(SELECTED_MODEL = 'MLP', SELECTED_DATASET = 'HIGHD', UNBALANCED = False, ABLATION = False)
     #1
    model_dict = m.MODELS[p.SELECTED_MODEL]

    model_dict['hyperparams']['task'] = params.CLASSIFICATION
    model_dict['hyperparams']['curriculum loss'] = False
    model_dict['hyperparams']['curriculum seq'] = False
    model_dict['hyperparams']['curriculum virtual'] = False

    model_dict['tag'] = utils.update_tag(model_dict)

    train_model_dict(model_dict, p)
   
    
   

    