import os
import torch
import torch.utils.data as utils_data
import numpy as np

class Parameters:
    def __init__(self, SELECTED_MODEL = 'VCNN', SELECTED_DATASET = 'HIGHD', UNBALANCED = False, ABLATION = False):
        # High Level Param
        self.SELECTED_MODEL = SELECTED_MODEL#'REGIONATTCNN3'#'VCNN'
        self.SELECTED_DATASET = SELECTED_DATASET
        self.UNBALANCED = UNBALANCED
        self.ABLATION = ABLATION
        
        self.ROBUST_PREDICTOR = True
        # Dataset Hyperparameters:
        self.DATASETS = {
            'HIGHD':{
                'abb_tr_ind':range(1,46),
                'abb_val_ind':range(46,51),
                'abb_te_ind':range(51,56),
                'tr_ind':range(1,51),
                'val_ind':range(51,56),
                'te_ind':range(56,61),
                'image_width': 256,
                'image_height': 32,
            }
        }
        self.IMAGE_HEIGHT = self.DATASETS[self.SELECTED_DATASET]['image_height']
        self.IMAGE_WIDTH = self.DATASETS[self.SELECTED_DATASET]['image_width']
        
        if self.ABLATION:
            self.TR_DATA_FILES = [ str(i).zfill(2)+'.h5' for i in self.DATASETS[self.SELECTED_DATASET]['abb_tr_ind']]
            self.VAL_DATA_FILES = [ str(i).zfill(2)+'.h5' for i in self.DATASETS[self.SELECTED_DATASET]['abb_val_ind']]
            self.TE_DATA_FILES = [ str(i).zfill(2)+'.h5' for i in self.DATASETS[self.SELECTED_DATASET]['abb_te_ind']]
        else:
            self.TR_DATA_FILES = [ str(i).zfill(2)+'.h5' for i in self.DATASETS[self.SELECTED_DATASET]['tr_ind']]
            self.VAL_DATA_FILES = [ str(i).zfill(2)+'.h5' for i in self.DATASETS[self.SELECTED_DATASET]['val_ind']]
            self.TE_DATA_FILES = [ str(i).zfill(2)+'.h5' for i in self.DATASETS[self.SELECTED_DATASET]['te_ind']]

        # Prediction Problem Hyperparameters:
        self.FPS = 5
        self.resume = False
        self.SEQ_LEN = 125
        self.RES = 5
        self.IN_SEQ_LEN = 1
        self.INPUT_HORIZON = 10
        self.loss_x_alpha = 0.05 #default should be 0.5
        self.vector_field_available = True
        self.TrainThePretrained = True ## False: no_grad for Pretrained models 
        self.input_normalization = False
        self.output_normalization = False

        # Metrics Hyperparameters:
        self.ACCEPTED_GAP = 0
        self.THR = 0.34

        # Training  Hyperparameters
        self.CUDA = True
        self.BATCH_SIZE = 128
        self.LR = 0.0005#  0.001
        self.LR_DECAY = 1
        self.LR_DECAY_EPOCH = 10
        self.NUM_EPOCHS = 200
        self.PATIENCE = 100
        self.TR_JUMP_STEP = 1 # was 10!

        if self.UNBALANCED:
            self.unblanaced_ext = 'U'
        else:
            self.unblanaced_ext = ''
        self.TRAIN_DATASET_DIR = '../../Dataset/Processed_highD/RenderedDataset/'
        self.TEST_DATASET_DIR = '../../Dataset/Processed_highD/RenderedDataset{}/'.format(self.unblanaced_ext)
        
        
        # CL Params
        self.MAX_PRED_TIME = self.SEQ_LEN-self.IN_SEQ_LEN + 1
        self.cl_step = 5
        self.start_seq_arange = np.arange(self.MAX_PRED_TIME-1,0, -1* self.cl_step)
        self.CL_EPOCH  = len(self.start_seq_arange)
        # self.START_SEQ_CL = np.concatenate((self.start_seq_arange, np.zeros((self.NUM_EPOCHS-len(self.start_seq_arange)))), axis = 0)
        # self.END_SEQ_CL = np.ones((self.NUM_EPOCHS))*(self.SEQ_LEN-self.IN_SEQ_LEN+1)
        
        # self.LOSS_RATIO_CL = np.concatenate((np.arange(self.CL_EPOCH)/self.CL_EPOCH, np.ones((self.NUM_EPOCHS-self.CL_EPOCH))), axis = 0)
        
        self.LOSS_RATIO_NoCL = 1

        self.MODELS_DIR = 'models/'
        self.RESULTS_DIR = 'results/'

        self.TABLES_DIR = self.RESULTS_DIR + 'tables/'
        self.FIGS_DIR = self.RESULTS_DIR + 'figures/'
        self.VIS_DIR = self.RESULTS_DIR + 'vis_data/'

        self.sample_image_DIR = self.RESULTS_DIR + 'train/'
        self.test_image_DIR = self.RESULTS_DIR + 'test/'
        self.last_model_DIR = self.RESULTS_DIR + 'model/'

        
        @property
        def SELECTED_DATASET(self):
            return self._SELECTED_DATASET
        
        @SELECTED_DATASET.setter
        def SELECTED_DATASET(self, val):
            self._SELECTED_DATASET = val
            self.IMAGE_HEIGHT = self.DATASETS[self._SELECTED_DATASET]['image_height']
            self.IMAGE_WIDTH = self.DATASETS[self._SELECTED_DATASET]['image_width']
            if self._ABLATION:
                self.TR_DATA_FILES = [ str(i).zfill(2)+'.h5' for i in self.DATASETS[self._SELECTED_DATASET]['abb_tr_ind']]
                self.VAL_DATA_FILES = [ str(i).zfill(2)+'.h5' for i in self.DATASETS[self._SELECTED_DATASET]['abb_val_ind']]
                self.TE_DATA_FILES = [ str(i).zfill(2)+'.h5' for i in self.DATASETS[self._SELECTED_DATASET]['abb_te_ind']]
            else:
                self.TR_DATA_FILES = [ str(i).zfill(2)+'.h5' for i in self.DATASETS[self._SELECTED_DATASET]['tr_ind']]
                self.VAL_DATA_FILES = [ str(i).zfill(2)+'.h5' for i in self.DATASETS[self._SELECTED_DATASET]['val_ind']]
                self.TE_DATA_FILES = [ str(i).zfill(2)+'.h5' for i in self.DATASETS[self._SELECTED_DATASET]['te_ind']]
            
            self.DATASET_DIR = '../../Dataset/Processed_highD/RenderedDataset{}/'.format(self.unblanaced_ext)
        
        
        @property 
        def ABLATION(self):
            return self._ABLATION
        
        @ABLATION.setter
        def ABLATION(self, val):
            self._ABLATION = val
            if self._ABLATION:
                self.TR_DATA_FILES = [ str(i).zfill(2)+'.h5' for i in self.DATASETS[self._SELECTED_DATASET]['abb_tr_ind']]
                self.VAL_DATA_FILES = [ str(i).zfill(2)+'.h5' for i in self.DATASETS[self._SELECTED_DATASET]['abb_val_ind']]
                self.TE_DATA_FILES = [ str(i).zfill(2)+'.h5' for i in self.DATASETS[self._SELECTED_DATASET]['abb_te_ind']]
            else:
                self.TR_DATA_FILES = [ str(i).zfill(2)+'.h5' for i in self.DATASETS[self._SELECTED_DATASET]['tr_ind']]
                self.VAL_DATA_FILES = [ str(i).zfill(2)+'.h5' for i in self.DATASETS[self._SELECTED_DATASET]['val_ind']]
                self.TE_DATA_FILES = [ str(i).zfill(2)+'.h5' for i in self.DATASETS[self._SELECTED_DATASET]['te_ind']]
        
        @property
        def UNBALANCED(self):
            return self._UNBALANCED

        @UNBALANCED.setter 
        def UNBALANCED(self, val):
            self._UNBALANCED = val
            if self._UNBALANCED:
                self.unblanaced_ext = 'U'
            else:
                self.unblanaced_ext = ''
            self.DATASET_DIR = '../../Dataset/Processed_highD/RenderedDataset{}/'.format(self.unblanaced_ext)
        
        
         


# Different Tasks
CLASSIFICATION = 0
REGRESSION = 1
DUAL = 2    
            
        
            

        

        