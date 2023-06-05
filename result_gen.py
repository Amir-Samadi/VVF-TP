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
from torchvision import transforms as tf
import utils
import tqdm
from copy import deepcopy
import torch.multiprocessing
from torchmetrics import MeanSquaredLogError
torch.multiprocessing.set_sharing_strategy('file_system')
from utils import NLL_loss
import matplotlib.colors as mcolors

# Set Random Seeds:
if torch.cuda.is_available():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.manual_seed_all(1)
else:
    device = torch.device("cpu")
torch.manual_seed(1)
torch.backends.cudnn.deterministic = True
np.random.seed(1)
random.seed(1)




def result_gen(model_dict, p, device):
    criterion = nn.MSELoss()
    displaySavePath = "display_results"
    if not os.path.exists(displaySavePath):
        os.makedirs(displaySavePath)

    print ('start loading dataset')
    dataFileName = 'AVs_VF'
    raw_data = None
    # Instantiate Dataset:                
    # DataSavePath = os.path.join('data',dataFileName, 'input_horizon_'+str(1)) 
    DataSavePath = os.path.join('/media/samadi_a/T7/reza/data3', dataFileName, 'input_horizon_'+str(10)) 
    dataset = CustomImageDataset(raw_data = raw_data ,
                                label_horizen = p.SEQ_LEN-p.IN_SEQ_LEN+2,
                                input_horizen = p.INPUT_HORIZON,
                                min_vehicle_frame_exist=p.INPUT_HORIZON,
                                p=p, DataSavePath=DataSavePath)
    train_length=int(0.7*len(dataset))
    test_length=len(dataset)-train_length
    tr_dataset = torch.utils.data.Subset(dataset, range(train_length))
    val_dataset = torch.utils.data.Subset(dataset, range(train_length, train_length+test_length ))
    tr_loader = utils_data.DataLoader(dataset = tr_dataset, shuffle = False, batch_size = p.BATCH_SIZE,
                                    drop_last= True, pin_memory= True, num_workers= 20)
    val_loader = utils_data.DataLoader(dataset = val_dataset, shuffle = False, batch_size = p.BATCH_SIZE,
                                        drop_last= True, pin_memory= True, num_workers= 20)
    
    print ('Dataset Len:', len(dataset))                

    try:
        batch_data = next(val_loader)
    except:
        data_iter = iter(val_loader)
        batch_data = next(data_iter) 
    
    batch_data = next(data_iter) 
    batch_data = next(data_iter) 
    batch_data = next(data_iter) 
    batch_data = next(data_iter) 
    batch_data = next(data_iter) 
    batch_data = next(data_iter) 
    batch_data = next(data_iter) 
    batch_data = next(data_iter) 
    batch_data = next(data_iter) 
    batch_data = next(data_iter) 
    batch_data = next(data_iter) 
    
    batch_data_occ = batch_data['occ']
    batch_data_mask = batch_data['mask']
    batch_data_v_x  = batch_data['v_x']
    batch_data_v_y  = batch_data['v_y']
    batch_label_x_fut = batch_data['x_fut']
    batch_label_y_fut = batch_data['y_fut']
    batch_data_x_init = batch_data['x_init']
    batch_data_y_init = batch_data['y_init']
    
    # predictions = {'CNN-LSTM-v4_VectorField_True_inputHorizon_1': None} 
    rmse_losses = dict()
    rmse_losses_y = dict()
    rmse_losses_x = dict()
    tags = []
    models = dict()

    for model_type in   [ 
                          'CNN-LSTM-v4',
                        # 'Prob-CNN-LSTM-v4'
                        ]:
        model_dict['hyperparams']['model type'] = model_type

        for vector_field in [False, True]:
            p.vector_field_available = vector_field


            for input_horizon in [int(1), int(10)]:
                p.INPUT_HORIZON = input_horizon

                tag = model_type + '_VectorField_' + str(p.vector_field_available)\
                +'_inputHorizon_' + str(p.INPUT_HORIZON)
                model_dict['tag'] = tag
                tags.append(tag)

                # Instantiate Model:
                models[tag] = model_dict['ref'](p.BATCH_SIZE, device, model_dict['hyperparams'], p).to(device).eval()
                
                print("loading the model's state_dict")
                models[tag].load_state_dict(torch.load(os.path.join('results' ,tag, 'model/', "BEST_VAL_LOSS_MODEL.pt")))

                
#############################################################################
#############################################################################
#############################################################################
#############################################################################
# ########################## rmse report 
#         # rmse initialization 
#         for tag in tags:
#             rmse_losses[tag] = (0, 0, 0, 0, 0)
#             rmse_losses_x[tag] = (0, 0, 0, 0, 0)
#             rmse_losses_y[tag] = (0, 0, 0, 0, 0)
#         print(tags)
#         data_lenght = len(val_loader)   
#         for idx, item in enumerate(val_loader): 
#             batch_data_occ = item['occ']
#             batch_data_mask = item['mask']
#             batch_data_v_x  = item['v_x']
#             batch_data_v_y  = item['v_y']
#             batch_label_x_fut = item['x_fut']
#             batch_label_y_fut = item['y_fut']           
#             for tag in tags:
#                 if "inputHorizon_10" in tag:
#                     input_horizon = 10
#                 else:
#                     input_horizon = 1

#                 if ('VectorField_True' in tag):
#                     if('VectorField_True' in tag and input_horizon==10):
#                         data_tuple = torch.cat((batch_data_occ,  batch_data_mask, batch_data_v_x*batch_data_mask, batch_data_v_y*batch_data_mask), 1).to(device)[:, :, 0:input_horizon]
#                     else:
#                         data_tuple = torch.cat((batch_data_occ,  batch_data_mask, batch_data_v_x, batch_data_v_y), 1).to(device)[:, :, 0:input_horizon]
#                 else:
#                     data_tuple = torch.cat((batch_data_occ, batch_data_mask), 1).to(device)[:, :, 0:input_horizon]    
#                 labels = torch.cat((batch_label_x_fut, batch_label_y_fut), 1).to(device)

#                 predictions_x = models[tag].eval()(data_tuple, labels)['ttlc_pred'][0].detach().cpu()
#                 predictions_y = models[tag].eval()(data_tuple, labels)['ttlc_pred'][1].detach().cpu()
#                 labels = labels.to('cpu')  
#         ####RMSE report 
#             # for tag in tags:
#                 pred = torch.stack((predictions_x, predictions_y)).permute(1,0,2)
#                 # print('alaki:  {}/{}'.format(idx, data_lenght))
#                 pred = torch.cumsum(pred,dim=-1)
#                 labels = torch.cumsum(labels,dim=-1)
#                 rmse1 = rmse_losses[tag][0] +  torch.sum((pred[:, :, :5 ] - labels[:, :, :5 ])**2) / (pred[:, :, :5 ].shape[0] * pred[:, :, :5 ].shape[2] * data_lenght)# RMSE for 1 sec prediction
#                 rmse2 = rmse_losses[tag][1] +  torch.sum((pred[:, :, :10] - labels[:, :, :10])**2) / (pred[:, :, :10].shape[0] * pred[:, :, :10].shape[2] * data_lenght)# RMSE for 2 sec prediction 
#                 rmse3 = rmse_losses[tag][2] +  torch.sum((pred[:, :, :15] - labels[:, :, :15])**2) / (pred[:, :, :15].shape[0] * pred[:, :, :15].shape[2] * data_lenght)# RMSE for 3 sec prediction 
#                 rmse4 = rmse_losses[tag][3] +  torch.sum((pred[:, :, :20] - labels[:, :, :20])**2) / (pred[:, :, :20].shape[0] * pred[:, :, :20].shape[2] * data_lenght)# RMSE for 4 sec prediction 
#                 rmse5 = rmse_losses[tag][4] +  torch.sum((pred[:, :, :25] - labels[:, :, :25])**2) / (pred[:, :, :25].shape[0] * pred[:, :, :25].shape[2] * data_lenght)# RMSE for 5 sec prediction 
#                 rmse_losses[tag] = (rmse1, rmse2, rmse3, rmse4, rmse5)
            

#                 pred = predictions_x
#                 labels_x = labels[:,0,:]
#                 pred = torch.cumsum(pred,dim=-1)
#                 rmse1_x = rmse_losses_x[tag][0] +  torch.sum((pred[:, :5 ] - labels_x[:, :5 ])**2) / (pred[:, :5 ].shape[0] * pred[:, :5 ].shape[1] * data_lenght)# RMSE for 1 sec prediction
#                 rmse2_x = rmse_losses_x[tag][1] +  torch.sum((pred[:, :10] - labels_x[:, :10])**2) / (pred[:, :10].shape[0] * pred[:, :10].shape[1] * data_lenght)# RMSE for 2 sec prediction 
#                 rmse3_x = rmse_losses_x[tag][2] +  torch.sum((pred[:, :15] - labels_x[:, :15])**2) / (pred[:, :15].shape[0] * pred[:, :15].shape[1] * data_lenght)# RMSE for 3 sec prediction 
#                 rmse4_x = rmse_losses_x[tag][3] +  torch.sum((pred[:, :20] - labels_x[:, :20])**2) / (pred[:, :20].shape[0] * pred[:, :20].shape[1] * data_lenght)# RMSE for 4 sec prediction 
#                 rmse5_x = rmse_losses_x[tag][4] +  torch.sum((pred[:, :25] - labels_x[:, :25])**2) / (pred[:, :25].shape[0] * pred[:, :25].shape[1] * data_lenght)# RMSE for 5 sec prediction 
#                 rmse_losses_x[tag] = (rmse1_x, rmse2_x, rmse3_x, rmse4_x, rmse5_x)


#                 pred = predictions_y
#                 labels_y = labels[:,1,:]
#                 pred = torch.cumsum(pred,dim=-1)
#                 rmse1_y = rmse_losses_y[tag][0] +  torch.sum((pred[:, :5 ] - labels_y[:, :5 ])**2) / (pred[:, :5 ].shape[0] * pred[:, :5 ].shape[1] * data_lenght)# RMSE for 1 sec prediction
#                 rmse2_y = rmse_losses_y[tag][1] +  torch.sum((pred[:, :10] - labels_y[:, :10])**2) / (pred[:, :10].shape[0] * pred[:, :10].shape[1] * data_lenght)# RMSE for 2 sec prediction 
#                 rmse3_y = rmse_losses_y[tag][2] +  torch.sum((pred[:, :15] - labels_y[:, :15])**2) / (pred[:, :15].shape[0] * pred[:, :15].shape[1] * data_lenght)# RMSE for 3 sec prediction 
#                 rmse4_y = rmse_losses_y[tag][3] +  torch.sum((pred[:, :20] - labels_y[:, :20])**2) / (pred[:, :20].shape[0] * pred[:, :20].shape[1] * data_lenght)# RMSE for 4 sec prediction 
#                 rmse5_y = rmse_losses_y[tag][4] +  torch.sum((pred[:, :25] - labels_y[:, :25])**2) / (pred[:, :25].shape[0] * pred[:, :25].shape[1] * data_lenght)# RMSE for 5 sec prediction 
#                 rmse_losses_y[tag] = (rmse1_y, rmse2_y, rmse3_y, rmse4_y, rmse5_y)


#         for tag in tags:
#             rmse_losses[tag] = torch.stack(rmse_losses[tag]).sqrt().tolist()
#             rmse_losses_x[tag] = torch.stack(rmse_losses_x[tag]).sqrt().tolist()
#             rmse_losses_y[tag] = torch.stack(rmse_losses_y[tag]).sqrt().tolist()
#             print("RMSE of model {}: 1 sec: {}, 2 sec: {}, 3 sec: {}, 4 sec: {}, 5 sec: {} \n".format(\
#                 tag, rmse_losses[tag][0], rmse_losses[tag][1], rmse_losses[tag][2], rmse_losses[tag][3], rmse_losses[tag][4]))

#             print("x-axis RMSE of model {}: 1 sec: {}, 2 sec: {}, 3 sec: {}, 4 sec: {}, 5 sec: {} \n".format(\
#                 tag, rmse_losses_x[tag][0], rmse_losses_x[tag][1], rmse_losses_x[tag][2], rmse_losses_x[tag][3], rmse_losses_x[tag][4]))

#             print("y-axis RMSE of model {}: 1 sec: {}, 2 sec: {}, 3 sec: {}, 4 sec: {}, 5 sec: {} \n".format(\
#                 tag, rmse_losses_y[tag][0], rmse_losses_y[tag][1], rmse_losses_y[tag][2], rmse_losses_y[tag][3], rmse_losses_y[tag][4]))

# # #############################################################################
# # # #############################################################################
############################################################################
#####################result display 
    predictions = {}
    for tag in tags:
        if "inputHorizon_10" in tag:
            input_horizon = 10
        else:
            input_horizon = 1

        if ('VectorField_True' in tag):
            if('VectorField_True' in tag and input_horizon==10):
                data_tuple = torch.cat((batch_data_occ,  batch_data_mask, batch_data_v_x*batch_data_mask, batch_data_v_y*batch_data_mask), 1).to(device)[:, :, 0:input_horizon]
            else:
                data_tuple = torch.cat((batch_data_occ,  batch_data_mask, batch_data_v_x, batch_data_v_y), 1).to(device)[:, :, 0:input_horizon]
        else:
            data_tuple = torch.cat((batch_data_occ, batch_data_mask), 1).to(device)[:, :, 0:input_horizon]    
        labels = torch.cat((batch_label_x_fut, batch_label_y_fut), 1).to(device)

        predictions[tag] = models[tag].eval()(data_tuple, labels)['ttlc_pred']

    for batch_idx in [88,85,82,79,75,71,68,65,62,58,54,50,46,42,38,34,30,26]: #final
    # for batch_idx in [16,15,14,13,12,11,9,7,5,2]: #final
    # for batch_idx in [54,58,62,65,68,71,74,76,78,79,80,81]:
    # for batch_idx in [30,33,36,39,42,46,48,52]:
    # for batch_idx in range(batch_data_occ.shape[0]):

        data_occ    = batch_data_occ[batch_idx].squeeze(0)    
        data_mask   = batch_data_mask[batch_idx].squeeze(0)   
        data_v_x    = batch_data_v_x[batch_idx].squeeze(0)    
        data_v_y    = batch_data_v_y[batch_idx].squeeze(0)    
        label_x_fut = batch_label_x_fut[batch_idx].squeeze(0) 
        label_y_fut = batch_label_y_fut[batch_idx].squeeze(0) 
        data_x_init = batch_data_x_init[batch_idx].squeeze(0) 
        data_y_init = batch_data_y_init[batch_idx].squeeze(0)

        fig, ax = plt.subplots(1)
        ax.imshow(data_occ[-1] + data_mask[-1])
        # x,y = np.meshgrid(np.arange(0, data_occ[-1].shape[1], 1), np.arange(0, data_occ[-1].shape[0], 1))
        # ax[1].axis('equal')
        # q = ax[1].quiver(x,y, data_v_x[-1].squeeze(0), data_v_y[-1].squeeze(0))
        
        ax.plot((data_x_init[0]+np.cumsum(np.insert(label_x_fut, 0,0)))*0.64,
                    (data_y_init[0]+np.cumsum(np.insert(label_y_fut, 0, 0)))*4,
                    '-', marker='.', markersize=8, linewidth=2.5)
        
        for tag in tags:
            ax.plot((data_x_init[0]+np.cumsum(np.insert(predictions[tag][0][batch_idx].cpu().detach().numpy(), 0,0)))*0.64,
                        (data_y_init[0]+np.cumsum(np.insert(predictions[tag][1][batch_idx].cpu().detach().numpy(), 0, 0)))*4,
                        '-', marker='.', markersize=8, linewidth=2.5)#label = tag
        ax.legend(["GT", "M1", "M2", "M3", "M4"], loc='upper right', fontsize=7)
        # ax.legend(loc='upper right', fontsize=7)
        
        custom_xlim = (0, data_occ[-1].shape[1])
        custom_ylim = (0, data_occ[-1].shape[0])
        # Setting the values for all axes.
        plt.setp(ax, xlim=custom_xlim, ylim=custom_ylim)

        plt.savefig(os.path.join(displaySavePath, 'data_num_{}'.format(batch_idx)+'.pdf'))
        plt.close(fig)
#############################################################################
#############################################################################
# #############################################################################
# #############################################################################

 


if __name__ == '__main__':
    
    
    p = params.Parameters(SELECTED_MODEL = 'VCNN', SELECTED_DATASET = 'HIGHD', UNBALANCED = False, ABLATION = False)

    #1
    model_dict = m.MODELS[p.SELECTED_MODEL]

    model_dict['hyperparams']['task'] = params.REGRESSION
    model_dict['hyperparams']['curriculum loss'] = False
    model_dict['hyperparams']['curriculum seq'] = False
    model_dict['hyperparams']['curriculum virtual'] = False

    model_dict['tag'] = utils.update_tag(model_dict)

    result_gen(model_dict, p, device)
    
    exit()
