# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 09:10:00 2025

@author: jmich
"""

import numpy as np

def merge_selected_npy_files(npy_files, output_file):
    """
    Loads selected .npy files, merges them along the zero axis, and saves the result.
    """
    
    arrays = [np.load(file).astype(np.float32) for file in npy_files]
    merged_array = np.concatenate(arrays, axis=0)

    np.save(output_file, merged_array)
    print(f"Sloučený soubor uložen jako: {output_file}")

if __name__ == "__main__":
    selected_files = []
    #for i in range(301, 401):
        #selected_files.append("AllDataY_Struct_P-1_"+str(i)+ ".npy")
    for i, j in [(1,100),(101,200),(201,300),(301,400),(401,500),(501,600),(601,700),(701,800), (801, 900), (901, 1000)]:
        selected_files.append("DATA/Merged_file_all_po100-orez/AllDataX_Struct_P-1_" + str(i) + "_"+str(j)+"_16x16_half"+ ".npy")
    output_file = "DATA/Merged_file_all_po100-orez/AllDataX_Struct_P-1_1_1000_8x8_half.npy"  # Název výstupního souboru
    merge_selected_npy_files(selected_files, output_file)
    #output_file = "AllDataY_Struct_P-1_301_400.npy"
    #merge_selected_npy_files(selected_files, output_file)
