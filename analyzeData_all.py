# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 22:09:40 2024

@author: jmich
"""
import loadData_all as ld
import numpy as np
import math
nacistData = True

vrstev = 3
for i in range(1,2):
    number_of_null = 0
    if i < 10:
        number_of_null = 3
    elif i < 100:
        number_of_null = 2
    elif i < 1000:
        number_of_null = 1
    files = ['Struct_0' + number_of_null * '0' + str(i)]
    print(files)
    new_file = 'Struct_P-1_' + str(i)
    AllDataX, AllDataY = ld.loadData(nacistData, files, new_file, vrstev)

