#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 09:52:26 2025

@author: michnjak
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
import time
import shutil
import matplotlib.pyplot as plt
from models_loss import Loss_WEIGHT_MSE, ResNetUNet3D, WeightedBCELoss, Masked_Accuracy, MLP_Mixer_arch, PhAINeuralNetwork, PhAINeuralNetwork1
from transform_load_data import Make_Dataloader
from torch.amp import autocast
from einops.layers.torch import Rearrange
from einops import rearrange
from prettytable import PrettyTable

def count_parameters(model):
    """
    Function that prints number of parameter of each layer in the model and return number of all trainable parameter in the model.
    """
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

def make_histogram(folder_name, targets, outputs):
    hist, bin_edges = torch.histogram(targets.to('cpu'), 20) # bins určuje počet sloupců
    #Transform into numpy
    hist = hist.numpy()
    bin_edges = bin_edges.numpy()

    #Drawing of histogram of true phase values
    plt.clf()
    plt.bar(bin_edges[:-1], hist, width=(bin_edges[1] - bin_edges[0]), edgecolor='black')
    plt.title('Histogram true value of phase')
    full_path = os.path.join(folder_name, "Histogram_true.png")
    
    plt.savefig(full_path)
    plt.close()
    hist, bin_edges = torch.histogram(outputs.to('cpu'), 20) # bins určuje počet sloupců
    #Transform into numpy
    hist = hist.numpy()
    bin_edges = bin_edges.numpy()

    #Drawing of histogram of predicted phase values
    plt.clf()
    plt.bar(bin_edges[:-1], hist, width=(bin_edges[1] - bin_edges[0]), edgecolor='black')
    plt.title('Histogram predicted value of phase')
    full_path = os.path.join(folder_name, "Histogram_pred.png")
    
    plt.savefig(full_path)
    plt.close()

def make_loss_plot(full_path, epoch, train_loss_list, val_loss_list):
    x_list = list(range(1, epoch+2))
    plt.clf()
    plt.plot(np.array(x_list), np.array(train_loss_list))
    plt.plot(np.array(x_list), np.array(val_loss_list), '-.')
    plt.savefig(full_path)
    plt.close()

def make_directories(file_name):
    #Creating directories in folder for saving predictions.
    os.makedirs(file_name + '/train_inputs', exist_ok=True)
    os.makedirs(file_name + '/train_prediction', exist_ok=True)
    os.makedirs(file_name + '/train_true', exist_ok=True)
    os.makedirs(file_name + '/val_inputs', exist_ok=True)
    os.makedirs(file_name + '/val_prediction', exist_ok=True)
    os.makedirs(file_name + '/val_true', exist_ok=True)


def train_model(model, train_loader, val_loader, criterion, optimizer, device, file_name, model_save_path, num_epochs=25):
    folder_name = file_name
    os.makedirs(folder_name, exist_ok=True)
    
    start_time = time.time()
    model.to(device)
    train_loss_list = []
    val_loss_list = []
    for epoch in range(num_epochs):
        #Training model
        model.train()
        train_running_loss = 0.0
        for inputs,norm_inputs, targets in train_loader:
            #Transforming inputs, norm_inputs and targets into device
            inputs = inputs.to(device)
            norm_inputs = norm_inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()

            #Mixed precision
            with autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(norm_inputs)
                loss = criterion(outputs, rearrange(targets,'B C H W D-> B C (H W D)', H = 16, W = 16), rearrange(norm_inputs,'B C H W D -> B C (H W D)', H = 16, W = 16))  # Předání inputs jako x_true pro CustomLoss
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_running_loss += loss.item() * inputs.size(0)
            
        train_running_loss = train_running_loss / len(train_loader.dataset)
        print("train_running_loss : ", train_running_loss )
        
        #Evaluation of model on train set
        model.eval()
        total_correct, total_valid_voxels = 0, 0
        train_loss = 0.0
        with torch.no_grad():
            for inputs, norm_inputs, targets in train_loader:
                inputs = inputs.to(device)
                norm_inputs = norm_inputs.to(device) 
                targets = targets.to(device)
                with autocast(device_type='cuda', dtype=torch.float16):
                    outputs = model(norm_inputs)
                    loss = criterion(outputs, rearrange(targets,'B C H W D-> B C (H W D)', H = 16, W = 16), rearrange(norm_inputs,'B C H W D -> B C (H W D)', H = 16, W = 16))  # Předání inputs jako x_true pro CustomLoss
                train_loss += loss.item() * inputs.size(0)
                total_correct, total_valid_voxels, train_accuracy = metric(outputs, rearrange(targets,'B C H W D -> B C (H W D)'), rearrange(norm_inputs,'B C H W D -> B C (H W D)', H = 16, W = 16), total_correct, total_valid_voxels)
            train_loss /= len(train_loader.dataset)
            #make_histogram(folder_name, targets, outputs)
            train_loss_list.append(train_loss)
        
        #Evaluation of model on validation set
        val_loss = 0.0
        total_correct, total_valid_voxels = 0, 0
        with torch.no_grad():
            for inputs, norm_inputs, targets in val_loader:
                inputs = inputs.to(device)
                norm_inputs = norm_inputs.to(device)
                targets = targets.to(device)
                with autocast(device_type='cuda', dtype=torch.float16):
                    outputs = model(norm_inputs)
                    loss = criterion(outputs, rearrange(targets,'B C H W D-> B C (H W D)', H = 16, W = 16), rearrange(norm_inputs,'B C H W D-> B C (H W D)', H = 16, W = 16))  # Předání inputs jako x_true pro CustomLoss
                val_loss += loss.item() * inputs.size(0)
                total_correct, total_valid_voxels, val_accuracy = metric(outputs,rearrange(targets,'B C H W D-> B C (H W D)', H = 16, W = 16), rearrange(norm_inputs,'B C H W D-> B C (H W D)', H = 16, W = 16), total_correct, total_valid_voxels)
        val_loss /= len(val_loader.dataset)
        val_loss_list.append(val_loss)
        
        #Printing loss and accuracy from training into console
        print(f'Epoch {epoch+1}/{num_epochs} - Train Running Loss: {train_running_loss:.4f}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train Accuracy: {train_accuracy :.4f}, Val Accuracy: {val_accuracy :.4f}\n')
        print(f'Elapsed time: {time.time() - start_time} s\n')

        #Printing loss and accuracy from training into text file
        full_path = os.path.join(folder_name, "results.txt")
        with open(full_path, 'a') as f:
            f.write(f'Epoch {epoch+1}/{num_epochs} - Train Running Loss: {train_running_loss:.4f}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train Accuracy: {train_accuracy :.4f}, Val Accuracy: {val_accuracy :.4f}\n')
            f.write(f'Elapsed time: {time.time() - start_time} s\n')

        #Using writer from tensorboard to add loss
        writer.add_histogram("Train loss", train_loss, epoch)
        writer.add_histogram("Validation loss", val_loss, epoch)
        
        #Saving model into folder
        if epoch % 100 == 0:
            torch.save(model.state_dict(), file_name + '/' + model_save_path)
        
        #Making loss plot
        #full_path = os.path.join(folder_name, "Train_loss.png")
        #make_loss_plot(full_path, epoch, train_loss_list, val_loss_list)
       

def predict(model, train_loader, val_loader, criterion, optimizer, device, file_name):
    model.to(device)
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        i = 0
        for inputs, norm_inputs, targets in train_loader:
            i = i + 1
            norm_inputs = norm_inputs.to(device)
            targets = targets.to(device)
            outputs = model(norm_inputs)
            loss = criterion(outputs, targets, norm_inputs)
            running_loss += loss.item() * inputs.size(0)
            np.save(file_name + '/train_inputs/' + str(i), inputs.numpy())
            np.save(file_name + '/train_prediction/' + str(i), outputs.to('cpu').numpy()*2*np.pi)
            np.save(file_name + '/train_true/' + str(i), targets.to('cpu').numpy()*2*np.pi)

        train_loss = running_loss / len(train_loader.dataset)
        print(train_loss)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        i = 0
        for inputs, norm_inputs, targets in val_loader:
            i = i + 1
            norm_inputs = norm_inputs.to(device)
            targets = targets.to(device)
            outputs = model(norm_inputs)
            loss = criterion(outputs, targets, norm_inputs)
            val_loss += loss.item() * inputs.size(0)
            np.save(file_name + '/val_inputs/' + str(i), inputs.numpy())
            np.save(file_name + '/val_prediction/' + str(i), outputs.to('cpu')*2*np.pi)
            np.save(file_name + '/val_true/' + str(i), targets.to('cpu')*2*np.pi)
            
        val_loss /= len(val_loader.dataset)
        print(val_loss)



if __name__ == '__main__':
    #File path to the amplitudes and phases 
    amplitude_file = '../Nove_data/NEW_DATA/Merged_file_all_po100-orez/AllDataX_Struct_1_1000_8x8.npy'
    phase_file = '../Nove_data/NEW_DATA/Merged_file_all_po100-orez/AllDataY_Struct_1_1000_8x8.npy'
    
    #Name of the folder to save model, results, prediction
    file_name = 'Resnet3D_1_1000k_vetsi_batchonly_Weight_MSE_P1_1e-4_log_normalized_each_sample_1pokus_paralel_mix'
    os.makedirs(file_name, exist_ok=True)
    
    #Name of the saved model in the folder
    path_to_load_model = 'unet3d_phase_supervised_1.pth'
    path_to_save_model = 'unet3d_phase_supervised_1.pth'
    
    #Creating writer to save pytorch logs
    writer = SummaryWriter(file_name + '/runs')

    #Coppying this code into folder
    source_file = os.path.abspath(__file__)
    shutil.copy2(source_file, file_name)
    
    #Parameters of the model and training
    num_epochs = 100
    loading_model = False
    lr = 1e-4
    divider = 1
    make_predition = False
    data_type="P1"
    sigmoid = True
    in_channels=1
    out_channels=1
    classification = False
    batch_size_train = 2000
    batch_size_val = 2000
    mode= "_"
    #only_half = True or False
    only_half = False

    #Creating of train and validation loader
    train_loader, val_loader = Make_Dataloader(amplitude_file, phase_file, divider, data_type, mode, batch_size_train, batch_size_val, only_half)
    
    #Creating model
    model = ResNetUNet3D(in_channels, out_channels, sigmoid)
    #model = MLP_Mixer_arch(in_channels, out_channels, sigmoid)
    #model_args = {'max_index':10, 'filters':32, 'kernel_size':3, 'cnn_depth': 2, 
    #              'dim': 1024, 'dim_exp':2048, 'dim_token_exp':512, 'mlp_depth':2,
    #              'reflections': 1205, 'data_size_mode':"full_8"}
    #data_size_mode = full_8, half_8, full_16, half_16
    #model = PhAINeuralNetwork(**model_args)
    
    #Printing model parameters in each layer
    print(count_parameters(model))

    #Creating scaler for mixed precission
    scaler = torch.cuda.amp.GradScaler()
    
    #Model paralelization
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    
    #Loading model
    if loading_model:
        model.load_state_dict(torch.load(file_name + '/' + path_to_load_model))

    #Setting of loss function
    if classification:
        criterion = WeightedBCELoss()
        metric = Masked_Accuracy
    else:
        criterion = Loss_WEIGHT_MSE()  # Změna na CustomLoss
    
    optimizer = optim.Adam(model.parameters(), lr)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #Training of model
    train_model(model, train_loader, val_loader, criterion, optimizer, device, file_name, path_to_save_model, num_epochs=num_epochs)
    
    #Saving model to directory
    torch.save(model.state_dict(), file_name + '/' + path_to_save_model)
    
    #Make prediction and save into folder
    if make_predition:
        predict(model, train_loader, val_loader, criterion, optimizer, device, file_name)
