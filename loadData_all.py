# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 17:44:18 2024

@author: jmich
"""

# -*- coding: utf-8 -*-


import concurrent.futures
import numpy as np
from gemmi import cif
import os


def loadData(nacistData, files, new_file, vrstev):
    global item, min_value_x, min_value_y, min_value_z, x_size, y_size, z_size, AllDataX, AllDataY
    if nacistData == True:
        #shift parameter const_x
        const_x = 40
        
        AllDataX = np.empty((0, 2*const_x+1, 2*const_x+1, 2*const_x-1))
        AllDataY = np.empty((0, 2*const_x+1, 2*const_x+1, 2*const_x-1))
        
        
        #Data processing function - centering cell dimensions to zero and storing values in new coordinates
        def process_item(item):
            DataArrayX = np.zeros([2*const_x + 1, 2*const_x + 1, const_x])
            DataArrayY = np.zeros([2*const_x + 1, 2*const_x + 1, const_x])
            if item.loop is not None and len(item.loop.tags) == 5:
                data = np.array(item.loop.values).reshape([-1, 5])
                for i in range(data.shape[0]):
                    if data[i, 0].astype(int) >= -const_x and data[i, 0].astype(int) <= const_x and data[i, 1].astype(int) >= -const_x and data[i, 1].astype(int) <= const_x and data[i, 2].astype(int) < const_x and data[i, 2].astype(int) >= 0:
                        DataArrayX[
                            data[i, 0].astype(int) + const_x, data[i, 1].astype(int) +const_x, data[i, 2].astype(
                                int)] = data[i, 3].astype(float)
                        DataArrayY[
                            data[i, 0].astype(int) + const_x, data[i, 1].astype(int) + const_x, data[i, 2].astype(
                                int)] = data[i, 4].astype(float)
            DataArrayX_flipped = np.flip(DataArrayX[:, :, 1:], (0,1,2))        
            DataArrayY_flipped = -np.flip(DataArrayY[:, :, 1:], (0,1,2))
            
            DataArrayX_true = DataArrayX
            DataArrayX_true[:, :, 0] = DataArrayX[:, :, 0]  + np.flip(DataArrayX[:, :, 0], axis=(0,1))

            DataArrayY_true = DataArrayY
            DataArrayY_true[:, :, 0] = DataArrayY[:, :, 0]  - np.flip(DataArrayY[:, :, 0], axis=(0,1))
            
            DataArrayX_true = np.concatenate((DataArrayX_flipped, DataArrayX_true), axis=2)
            DataArrayY_true = np.concatenate((DataArrayY_flipped, DataArrayY_true), axis=2)
            DataArrayY_true = np.where(DataArrayY_true<0, DataArrayY_true+2*np.pi, DataArrayY_true)

            return DataArrayX_true, DataArrayY_true

        num_threads = 16
        print("Number of threads: ", num_threads)

        AllDataX = []
        AllDataY = []
        i = 0
        for file in files: 
            directory_file = "/mnt/lustre/helios-shared/FZU_ML_crystal/TrainingData_P-1/"

            i += 1
            print(directory_file + file + '.cif')
            doc = cif.read_file(directory_file + file + '.cif')
            num = 0
            for block in doc:
                num = num + 1
                print(num)
                with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
                    results = list(executor.map(process_item, block))

                DataArrayXC = np.zeros([2*const_x + 1, 2*const_x + 1, 2*const_x -1])
                DataArrayYC = np.zeros([2*const_x + 1, 2*const_x + 1, 2*const_x - 1])
                for DataArrayX, DataArrayY in results:
                    DataArrayXC = DataArrayXC + DataArrayX
                    DataArrayYC = DataArrayYC + DataArrayY

                AllDataX.append(DataArrayXC)
                AllDataY.append(DataArrayYC)
                
        AllDataX = np.stack(AllDataX, axis=0)
        AllDataY = np.stack(AllDataY, axis=0)
        
        
        np.save("AllDataX_" + new_file, AllDataX.astype(np.float32))
        np.save("AllDataY_" + new_file, AllDataY.astype(np.float32))
        print(AllDataX.shape)
        print(AllDataY.shape)

    else:
        AllDataX = np.load("AllDataX_" + new_file + ".npy")
        AllDataY = np.load("AllDataY_" + new_file + ".npy")
    return AllDataX, AllDataY
