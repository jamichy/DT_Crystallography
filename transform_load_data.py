#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 21:33:24 2025

@author: michnjak
"""
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset, Subset


class PerSampleMinMaxNormalize:
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        min_val = tensor.min()
        max_val = tensor.max()
        return (tensor - min_val) / (max_val - min_val + 1e-8)

class MyDataset(Dataset):
    def __init__(self, amplitude_file, phase_file, transform=None, data_type='P1', only_half=False):
        if data_type == 'P1':
            self.phases = np.load(phase_file)/(2*np.pi)
        else:
            self.phases = (np.load(phase_file)%(2*np.pi))/np.pi
            self.phases[self.phases > 0.6] = 1.000

        self.amplitudes = np.log(np.sqrt(np.load(amplitude_file)) + 1)
        if only_half:
            self.amplitudes[:, :, :self.amplitudes.shape[2]//2, 0] = self.amplitudes[:, :, :self.amplitudes.shape[2]//2, 0]*0.0

        self.transform = transform
        if self.transform:
            self.normalized_amplitudes = np.stack([self.transform(torch.tensor(sample)).numpy() 
                                                  for sample in self.amplitudes], axis=0)
        else:
            self.normalized_amplitudes = self.amplitudes
        print(f"Loaded dataset: amplitudes shape {self.normalized_amplitudes.shape}, phases shape {self.phases.shape}")

    def __len__(self):
        return self.amplitudes.shape[0]

    def __getitem__(self, idx):
        amplitude = self.amplitudes[idx]
        normalized_amplitudes = self.normalized_amplitudes[idx]
        phase = self.phases[idx]
        amplitude = torch.tensor(amplitude, dtype=torch.float32).unsqueeze(0)
        normalized_amplitudes = torch.tensor(normalized_amplitudes, dtype=torch.float32).unsqueeze(0)
        phase = torch.tensor(phase, dtype=torch.float32).unsqueeze(0)
        return amplitude, normalized_amplitudes, phase

def Make_Dataloader(amplitude_file, phase_file, divider, data_type="P1", mode="_",
                    batch_size_train = 80, batch_size_val = 80, only_half = False):
    transform = PerSampleMinMaxNormalize()
    dataset = MyDataset(amplitude_file, phase_file, transform, data_type, only_half)

    
    total_samples = len(dataset)//divider
    indices = np.arange(total_samples)
    if mode == "80/20":
        train_size = int(0.8 * total_samples)  # 80 % train data
        val_size = total_samples - train_size   # 20 % validation data

        #First part - train indices, second part validation indices
        train_indices = list(range(0, train_size))
        val_indices = list(range(train_size, total_samples))
    else:
        train_indices = indices[indices % 5 != 4]
        val_indices = indices[indices % 5 == 4]

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    train_loader = DataLoader(train_dataset, batch_size_train, shuffle=True, num_workers=1)
    val_loader = DataLoader(val_dataset, batch_size_val, shuffle=True, num_workers=1)
    return train_loader, val_loader