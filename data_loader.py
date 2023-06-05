import os
from scipy import io
import numpy as np
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import torch

from torch.nn.functional import normalize

class dataType:
        def __init__(self):
            self.ID, self.frameNum , self.x_fut , self.y_fut , self.mask ,\
                 self.v_x , self.v_y , self.occ, self.init_x, self.init_y = ([] for i in range(10))

class CustomImageDataset(Dataset):
    def __init__(self, raw_data, label_horizen=None, input_horizen=None,
                min_vehicle_frame_exist=None, transform=False, target_transform=False, p=None, DataSavePath=None):
        
        self.raw_data = raw_data
# raw_Data format
#     |frame 1	| car 1 : ID, X_fut(samples), Y_fut(samples), Mask(nx*ny) | car2: ID, X_fut, Y_fut, Mask | ...
#                 |v_x(nx*ny), v_y(nx*ny), occ(nx*ny)
#     |frame 2 ...   
        self.label_horizen = label_horizen
        self.input_horizen = input_horizen
        self.min_vehicle_frame_exist = min_vehicle_frame_exist
        self.p = p  
        self.transform = transform
        self.target_transform = target_transform
        self.data = dataType()
        self.DataSavePath = DataSavePath 

        
    def prepare(self):
        cntr=0
        for idx, frame in enumerate(self.raw_data):
            try: 
                len(frame[0])
            except:
                continue
            
            if len(frame[0]) != 2:
                # print("something is wrong at frame: ",idx)
                continue
            for veh_idx, vehicle in enumerate(frame[0][0]):              
                try:
                    (vehicle[1].shape[0] < self.label_horizen-1) or (vehicle[4] < self.min_vehicle_frame_exist)
                except:
                    # print("something is wrong at vehicle: ",veh_idx, "frame: ",idx)
                    # print(vehicle)
                    continue

                if(vehicle[1].shape[0] < self.label_horizen-1) or (vehicle[4] < self.min_vehicle_frame_exist):
                    # print("something is wrong at vehicle: ",veh_idx, "frame: ",idx)
                    continue
                else:
                        self.data.frameNum = idx
                        self.data.ID.append(vehicle[0])
                        # self.data.x_fut.append((vehicle[5][0:label_horizen-1:p.RES]/0.64)*p.RES)
                        # self.data.y_fut.append((vehicle[6][0:label_horizen-1:p.RES]/4)*p.RES)                        
                        self.data.init_x.append(vehicle[1][0:self.label_horizen-1:self.p.RES]/0.64)
                        self.data.init_y.append(vehicle[2][0:self.label_horizen-1:self.p.RES]/4)

                        dx_fut = self.data.init_x[-1][1:] - self.data.init_x[-1][0:-1]
                        dy_fut = self.data.init_y[-1][1:] - self.data.init_y[-1][0:-1]
                        
                        self.data.x_fut.append(np.insert(dx_fut,-1,dx_fut[-1]))
                        self.data.y_fut.append(np.insert(dy_fut,-1,dy_fut[-1]))
                        
                        # self.data.mask.append(vehicle[0,3])
                        
                        mask_temp, v_x_temp, v_y_temp, occ_temp = [],[],[],[]
                        for frame_ii in range(idx-self.input_horizen+1,idx+1):
                            for jj, horizon_vehicle in enumerate(self.raw_data[frame_ii][0][0]):
                                if(horizon_vehicle[0] == vehicle[0]):
                                    mask_temp.append(horizon_vehicle[3])
                                    v_x_temp.append(self.raw_data[frame_ii][0][1][0])
                                    v_y_temp.append(self.raw_data[frame_ii][0][1][1])
                                    occ_temp.append(self.raw_data[frame_ii][0][1][2])
                                    break

                        self.data.mask.append(mask_temp)
                        self.data.occ.append(occ_temp)
                        self.data.v_x.append(np.multiply(v_x_temp,1))         # Reza: multplying vx by scalar 0/1 to alter VF effect
                        self.data.v_y.append(np.multiply(v_y_temp,1))         # Reza: multplying vy by scalar 0/1 to alter VF effect
                        # self.data.v_x.append(np.multiply(v_x_temp,mask_temp))   # Reza: multplying vx by vehicle's mask to alter VF effect
                        # self.data.v_y.append(np.multiply(v_y_temp,mask_temp))   # Reza: multplying vy by vehicle's mask to alter VF effect
                      
                        # #same occ, v_x, and v_y for all vehicles in a frame
                        # self.data.v_x.append(frame[0][1,0])
                        # self.data.v_y.append(frame[0][1,1])
                        # self.data.occ.append(np.flip(frame[0][1,2], axis=0))
                
                if (len(self.data.occ[0])==self.input_horizen):
                    np.savez_compressed(os.path.join(self.DataSavePath, str(cntr)+'.npz'), occ=self.data.occ,\
                        mask=self.data.mask, v_x=self.data.v_x, v_y=self.data.v_y, x_fut=self.data.x_fut, y_fut=self.data.y_fut,
                        init_x = self.data.init_x, init_y=self.data.init_y)
                    self.data = dataType()
                    cntr += 1

                


                # self.data.occ   = torch.FloatTensor(np.array(self.data.occ))   
                # self.data.v_y   = torch.FloatTensor(np.array(self.data.v_y))  # normalize(torch.FloatTensor(np.array(self.data.v_y)), p=1) 
                # self.data.v_x   = torch.FloatTensor(np.array(self.data.v_x))  # normalize(torch.FloatTensor(np.array(self.data.v_x)), p=1)
                # # self.data.v_y   = normalize(torch.FloatTensor(np.array(self.data.v_y)), p=1) 
                # # self.data.v_x   = normalize(torch.FloatTensor(np.array(self.data.v_x)), p=1) 
                # self.data.mask  = torch.FloatTensor(np.array(self.data.mask))
                # self.data.x_fut = torch.FloatTensor(np.array(self.data.x_fut))
                # self.data.y_fut = torch.FloatTensor(np.array(self.data.y_fut))
                # self.data.init_x = torch.FloatTensor(np.array(self.data.init_x))
                # self.data.init_y = torch.FloatTensor(np.array(self.data.init_y))


    def __len__(self):
        return len(self.DataSavePath)


    def data_display(self, data, frameNumber=None):
        occ = data.occ[frameNumber]
        fig, ax = plt.subplots(4,1)
        ax[0].imshow(occ)
        ax[0].invert_yaxis()
        x,y = np.meshgrid(np.arange(0, occ.shape[1], 1), np.arange(0, occ.shape[0], 1))
        q = ax[1].quiver(x,y,data.v_x[frameNumber],data.v_y[frameNumber])
        ax[1].axis('equal')
        plt.show()
        
    def __len__(self):
        return len(os.listdir(self.DataSavePath))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        data = np.load(os.path.join(self.DataSavePath, str(idx)+'.npz'))
        
        # if data['occ'].shape != (1,10,32,256):
        #     print (data['occ'].shape)
        #     # print (data['occ'])
        #     print(idx)
        # if data['mask'].shape != (1,10,32,256):
        #     print (data['mask'].shape)
        #     # print (data['mask'])
        #     print(idx)
        # if data['v_x'].shape != (1,10,32,256):
        #     print (data['v_x'].shape)
        #     # print (data['v_x'])
        #     print(idx)
        # if data['v_y'].shape != (1,10,32,256):
        #     print (data['v_y'].shape)
        #     # print (data['v_y'])
        #     print(idx)
        # if data['x_fut'].shape != (1,25):
        #     print (data['x_fut'].shape)
        #     # print (data['x_fut'])
        #     print(idx)
        # if data['y_fut'].shape != (1,25):
        #     print (data['y_fut'].shape)
        #     # print (data['y_fut'])
        #     print(idx)
        # if data['init_x'].shape != (1,25):
        #     print (data['init_x'].shape)
        #     # print (data['init_x'])
        #     print(idx)
        # if data['init_y'].shape != (1,25):
        #     print (data['init_y'].shape)
        #     # print (data['init_y'])
        #     print(idx)
            
        

        return {
            #image
            "occ" : torch.FloatTensor(data['occ']),
            "mask" : torch.FloatTensor(data['mask']),

            # "v_x" : torch.FloatTensor(np.multiply(data['v_x'], data['mask'])),
            # "v_y" : torch.FloatTensor(np.multiply(data['v_y'], data['mask'])),

            "v_x" : torch.FloatTensor(data['v_x']),
            "v_y" : torch.FloatTensor(data['v_y']),

            #label which is dx actually
            "x_fut" : torch.FloatTensor(data['x_fut']),
            "y_fut" : torch.FloatTensor(data['y_fut']),
            #initial x and y for each data 
            "x_init" : torch.FloatTensor(data['init_x']),
            "y_init" : torch.FloatTensor(data['init_y']), 
            "index" : idx
            }