# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 22:15:57 2024
@author: jmich
"""

import numpy as np

"""
Amplitudes, or phases with index [0,0,0] are stored at position [30,30,29]
"""

sirka_x = 8
sirka_y = 8
sirka_z = 8

param_1 = 30
#mode - parameter {"full": full sphere, "half": hemisphere, "half+": hemisphere+}
mode = "full"

for i, j in [(1, 100),(101, 200),(201, 300),(301, 400),(401, 500),(501, 600), (601, 700), (701, 800), (801, 900), (901, 1000)]:
    loading_x_path = 'DATA/Merged_file_all_po100/AllDataX_Struct_P-1_' + str(i) + '_' + str(j)+'.npy'
    loading_y_path = 'DATA/Merged_file_all_po100/AllDataY_Struct_P-1_' + str(i) + '_' + str(j)+'.npy'
    saving_x_path = 'DATA/Merged_file_all_po100-orez/AllDataX_Struct_P-1_' + str(i) + '_' + str(j)+'_8x8_half.npy'
    saving_y_path = 'DATA//Merged_file_all_po100-orez/AllDataY_Struct_P-1_' + str(i) + '_' + str(j)+'_8x8_half.npy'
    
    print("Loading")
    loading_file_X = np.load(loading_x_path)
    loading_file_X = loading_file_X.astype(dtype=np.float32)
    print("Loaded")
    loading_file_Y = np.load(loading_y_path)
    loading_file_Y = loading_file_Y.astype(dtype=np.float32)

    if mode == "full":
        print("Full")
        np.save(saving_x_path, loading_file_X[:, param_1-sirka_x:param_1+sirka_x, param_1-sirka_y: param_1+sirka_y, param_1-1-sirka_z:param_1-1+sirka_z])
        np.save(saving_y_path, loading_file_Y[:, param_1-sirka_x:param_1+sirka_x, param_1-sirka_y:param_1+sirka_y, param_1-1-sirka_z:param_1-1+sirka_z])
    elif mode == "half+":
        print("Half+")
        np.save(saving_x_path, loading_file_X[:, param_1-sirka_x:param_1+sirka_x, param_1-sirka_y: param_1+sirka_y, param_1-1:param_1+sirka_z])
        np.save(saving_y_path, loading_file_Y[:, param_1-sirka_x:param_1+sirka_x, param_1-sirka_y:param_1+sirka_y, param_1:param_1-1+sirka_z])
    else:
        print("Half")
        loading_file_X = loading_file_X[:, param_1-sirka_x:param_1+sirka_x, param_1-sirka_y: param_1+sirka_y, param_1-1:param_1-1+sirka_z]
        loading_file_Y = loading_file_Y[:, param_1-sirka_x:param_1+sirka_x, param_1-sirka_y:param_1+sirka_y, param_1-1:param_1-1+sirka_z]
        loading_file_X[:,param_1-sirka_x:param_1+sirka_x, param_1-sirka_y: param_1 ,0] *= 0.0
        loading_file_Y[:,param_1-sirka_x:param_1+sirka_x, param_1-sirka_y: param_1 ,0] *= 0.0
        np.save(saving_x_path, loading_file_X)
        np.save(saving_y_path, loading_file_Y)
        