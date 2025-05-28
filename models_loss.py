#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 19:37:10 2025

@author: michnjak
"""

import torch
import torch.nn as nn
from einops.layers.torch import Rearrange, Reduce


def Masked_Accuracy(outputs, targets, weights, total_correct, total_valid_voxels):
    predictions = (outputs >= 0.5).float()
    
    # Compute correct predictions
    correct = (predictions == targets).float()
    
    # Apply mask for weights > 0
    mask = (weights > 0).float()
    masked_correct = correct * mask

    # Accumulate totals
    total_correct = masked_correct.sum().item()
    total_valid_voxels = mask.sum().item()
    if total_valid_voxels > 0:
        accuracy = total_correct / total_valid_voxels
    else:
        accuracy = 0
    return total_correct, total_valid_voxels, accuracy


class Loss_WEIGHT_MSE(nn.Module):
    def __init__(self):
        super(Loss_WEIGHT_MSE, self).__init__()

    def forward(self, y_true, y_pred, x_true, arg="_"):  # Změna výchozího arg na 0 pro periodičnost
        #two_targets_0_1 for two possible targets for phase in interval (0, 1)
        if arg == "two_targets_0_1":
            y_target = y_true / 2 + 0.25
            angle_diff = torch.abs(y_pred - y_target)
            angle_diff_plus = torch.abs(y_pred + 0.5 - y_target)
            angle_diff_minus = torch.abs(y_pred - 0.5 - y_target)
            angle_difference = torch.min(torch.min(angle_diff_plus, angle_diff), angle_diff_minus)
        elif arg == "period_0_1":
            #Periodic corection for range (0,1)
            angle_difference = y_true - y_pred
            angle_difference = torch.where(angle_difference > 0.5, angle_difference - 1, angle_difference)
            angle_difference = torch.where(angle_difference < -0.5, angle_difference + 1, angle_difference)
        else:
            #without periodicity
            angle_difference = y_true - y_pred
        
        not_zero_mask = x_true >= 0.0
        list_of_sumation = [i+1 for i in range(x_true.dim() - 1)]
        result = angle_difference.pow(2) * x_true * not_zero_mask
        batch_sum_amp = torch.sum(x_true * not_zero_mask, list_of_sumation, keepdim=False)
        lossAvg = torch.sum(torch.sum(result, list_of_sumation, keepdim=False) / batch_sum_amp, 0, keepdim=False) / batch_sum_amp.shape[0]
        return lossAvg

class WeightedBCELoss(nn.Module):
    def __init__(self, epsilon=1e-8):
        super(WeightedBCELoss, self).__init__()
        self.epsilon = epsilon
        
    def forward(self, outputs, targets, weights):
        """
        outputs: output of the model after sigmoid, shape [B, 1, D, H, W]
        targets: ground truth (0 or 1), shape [B, 1, D, H, W]
        weights: shape [B, 1, D, H, W]
        """
        loss = torch.nn.functional.binary_cross_entropy_with_logits(outputs, targets)
        list_of_sumation = [i for i in range(inputs.dim())]
        return torch.sum(weights*loss, list_of_sumation, keepdim=False)/torch.sum(weights, list_of_sumation, keepdim=False)
        
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm3d(out_channels)
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample:
            identity = self.downsample(x)

        out += identity
        return self.relu(out)


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.resblock = ResidualBlock(out_channels, out_channels)

    def forward(self, x):
        x = self.up(x)
        x = self.resblock(x)
        return x


class MLPMixer3D(nn.Module):
    def __init__(self, num_patches=16*16*16, num_channels=64, token_mlp_dim=512, channel_mlp_dim=128):
        super().__init__()
        self.layernorm1 = nn.LayerNorm(num_channels)
        self.token_mixing = nn.Sequential(
            nn.Linear(num_patches, token_mlp_dim),
            nn.GELU(),
            nn.Linear(token_mlp_dim, num_patches)
        )
        
        self.layernorm2 = nn.LayerNorm(num_channels)
        self.channel_mixing = nn.Sequential(
            nn.Linear(num_channels, channel_mlp_dim),
            nn.GELU(),
            nn.Linear(channel_mlp_dim, num_channels)
        )

    def forward(self, x):
        # x shape: [batch, D, H, W, C] => [batch, N, C] where N = D*H*W
        x = x.permute(0, 2, 3, 4, 1)

        B, D, H, W, C = x.shape
        x = x.view(B, D * H * W, C)

        # Token-mixing MLP
        y = self.layernorm1(x)
        y = y.transpose(1, 2)  # [B, C, N]
        y = self.token_mixing(y)
        y = y.transpose(1, 2)
        x = x + y  # Residual connection

        # Channel-mixing MLP
        y = self.layernorm2(x)
        y = self.channel_mixing(y)
        x = x + y  # Residual connection

        # Reshape back to original shape
        x = x.view(B, D, H, W, C)
        x = x.permute(0, 4, 1, 2, 3)
        return x



class ResNetUNet3D(nn.Module):
    """
    3D ResNet-like encoder-decoder plus 2 MLP Mixers
    """
    def __init__(self, in_channels, out_channels, sigmoid = False, num_patches=16*16*16, num_channels=64, token_mlp_dim=512, channel_mlp_dim=128):
        super(ResNetUNet3D, self).__init__()

        self.sigmoid = sigmoid

        self.enc1 = ResidualBlock(in_channels, 64)
        self.enc2 = ResidualBlock(64, 128, stride=2)
        self.enc3 = ResidualBlock(128, 256, stride=2)

        self.bottleneck = ResidualBlock(256, 512, stride=2)

        self.dec3 = UpBlock(512, 256)
        self.dec2 = UpBlock(256, 128)
        self.dec1 = UpBlock(128, 64)
        
        self.MLP_mixer_1 = MLPMixer3D()
        self.MLP_mixer_2 = MLPMixer3D()

        self.out_conv = nn.Conv3d(64, out_channels, kernel_size=1)
        if self.sigmoid:
            self.final_act = nn.Sigmoid()  # Nebo Softmax, podle účelu

    def forward(self, x):
        x1 = self.enc1(x)  # -> 64
        x2 = self.enc2(x1)  # -> 128
        x3 = self.enc3(x2)  # -> 256

        x = self.bottleneck(x3)  # -> 512
        x = self.dec3(x)  # -> 256
        x = self.dec2(x)  # -> 128
        x = self.dec1(x)  # -> 64

        x = self.MLP_mixer_1(x)
        x = self.MLP_mixer_2(x)
        x = self.out_conv(x)  # -> out_channels
        if self.sigmoid:
            x = self.final_act(x)
        return x


class ResNet3D(nn.Module):
    """
    3D ResNet-like encoder-decoder
    """
    def __init__(self, in_channels, out_channels, sigmoid = False, num_patches=16*16*16, num_channels=64, token_mlp_dim=512, channel_mlp_dim=128):
        super(ResNet3D, self).__init__()

        self.sigmoid = sigmoid

        self.enc1 = ResidualBlock(in_channels, 64)
        self.enc2 = ResidualBlock(64, 128, stride=2)
        self.enc3 = ResidualBlock(128, 256, stride=2)

        self.bottleneck = ResidualBlock(256, 512, stride=2)

        self.dec3 = UpBlock(512, 256)
        self.dec2 = UpBlock(256, 128)
        self.dec1 = UpBlock(128, 64)
        
        
        self.out_conv = nn.Conv3d(64, out_channels, kernel_size=1)
        if self.sigmoid:
            self.final_act = nn.Sigmoid()  # Nebo Softmax, podle účelu

    def forward(self, x):
        x1 = self.enc1(x)  # -> 64
        x2 = self.enc2(x1)  # -> 128
        x3 = self.enc3(x2)  # -> 256

        x = self.bottleneck(x3)  # -> 512
        x = self.dec3(x)  # -> 256
        x = self.dec2(x)  # -> 128
        x = self.dec1(x)  # -> 64

        x = self.out_conv(x)  # -> out_channels
        if self.sigmoid:
            x = self.final_act(x)
        return x




class MLP_Mixer_arch(nn.Module):
    def __init__(self, in_channels, out_channels, sigmoid = False, number_of_mixers = 5, num_patches=16*16*16, num_channels=64, token_mlp_dim=512, channel_mlp_dim=128):
        super(MLP_Mixer_arch, self).__init__()

        self.sigmoid = sigmoid
        self.enc1 = ResidualBlock(in_channels, 64)
        
        self.MLP_mixers = nn.Sequential(
            *[MLPMixer3D() for i in range(number_of_mixers)]
        )

        self.out_conv = nn.Conv3d(64, out_channels, kernel_size=1)
        if self.sigmoid:
            self.final_act = nn.Sigmoid()

    def forward(self, x):
        x = self.enc1(x)
        x = self.MLP_mixers(x)
        x = self.out_conv(x)
        if self.sigmoid:
            x = self.final_act(x)
        return x


class ConvolutionalBlock(nn.Module):
    def __init__(self, filters, kernel_size, padding):
        super().__init__()

        self.act = nn.GELU()

        self.conv1 = nn.Conv3d(filters, filters, kernel_size = kernel_size, padding = padding)
        self.conv2 = nn.Conv3d(filters, filters, kernel_size = kernel_size, padding = padding)

        self.norm1 = nn.GroupNorm(filters, filters)
        self.norm2 = nn.GroupNorm(filters, filters)

    def forward(self, x):

        identity = x

        x = self.conv1(x)
        x = self.act(x)
        x = self.norm1(x)

        x = self.conv2(x)
        x = self.act(x)
        x = self.norm2(x)

        x = x + identity
        return x

class MLPLayer(nn.Module):
    def __init__(self, token_nr, dim, dim_exp, mix_type):
        super().__init__()

        self.act    = nn.GELU()

        self.norm1  = nn.GroupNorm(token_nr, token_nr)

        if mix_type == 'token':
            self.layer1 = nn.Conv1d(kernel_size = 1, in_channels = token_nr, out_channels = dim_exp)
            self.layer2 = nn.Conv1d(kernel_size = 1, in_channels = dim_exp, out_channels = token_nr)
        else:
            self.layer1 =  nn.Linear(dim , dim_exp)
            self.layer2 =  nn.Linear(dim_exp, dim)

        self.mix_type = mix_type

    def forward(self, x):
        identity = x

        x = self.norm1(x)

        x = self.layer1(x)
        x = self.act(x)
        x = self.layer2(x)

        x = x + identity

        return x



class PhAINeuralNetwork(nn.Module):
    """
    Change of PhAI from paper - input and output have same dimensionality. Added fully conected layer as the final layer.
    """
    def __init__(self, *, max_index, filters, kernel_size, cnn_depth, dim, dim_exp, dim_token_exp, mlp_depth, reflections, data_size_mode):
        super().__init__()
        self.data_size_mode = data_size_mode
        if data_size_mode == "full_8":
            hkl = [16, 16, 16]
        elif data_size_mode == "half_8":
            hkl = [16,16,8]
        elif data_size_mode == "full_16":
            hkl = [32,32,32]
        elif data_size_mode == "half_16":
            hkl = [32,32,16]
        mlp_token_nr  = filters
        padding       = int((kernel_size - 1) / 2)
        self.sigmoid = nn.Sigmoid()
        self.net_a = nn.Sequential(
            #Rearrange('b x y z  -> b 1 x y z '),

            nn.Conv3d(1, filters, kernel_size = kernel_size, padding=padding),
            nn.GELU(),
            nn.GroupNorm(filters, filters)
        )

        self.net_p = nn.Sequential(
            Rearrange('b x y z  -> b 1 x y z '),

            nn.Conv3d(1, filters, kernel_size = kernel_size, padding=padding),
            nn.GELU(),
            nn.GroupNorm(filters, filters)
        )

        self.net_convolution_layers = nn.Sequential(
            *[nn.Sequential(
                ConvolutionalBlock(filters, kernel_size = kernel_size, padding = padding),
            ) for _ in range(cnn_depth)],
        )

        self.net_projection_layer = nn.Sequential(
            Rearrange('b c x y z  -> b c (x y z)'),
            nn.Linear(hkl[0]*hkl[1]*hkl[2], dim),
        )

        self.net_mixer_layers = nn.Sequential(
            *[nn.Sequential(
                MLPLayer(mlp_token_nr, dim, dim_token_exp, 'token'),
                MLPLayer(mlp_token_nr, dim, dim_exp      , 'channel'),
            ) for _ in range(mlp_depth)],
            nn.LayerNorm(dim),
        )

        self.net_output = nn.Sequential(
            Rearrange('b t x -> b x t'),
            nn.AdaptiveAvgPool1d(1),
            Rearrange('b x 1 -> b x'),

            nn.Linear(dim, hkl[0]*hkl[1]*hkl[2]),
            Rearrange('b x -> b 1 x'),
            Rearrange('b 1 (x y z) -> b 1 x y z', x = hkl[0], y=hkl[1]),
        )
        

    def forward(self, input_amplitudes):

        a = self.net_a(input_amplitudes)
        
        x = self.net_convolution_layers(a)

        x = self.net_projection_layer(x)

        x = self.net_mixer_layers(x)

        phases = self.net_output(x)
        #sigmoid_phases = self.sigmoid(phases)
        return phases
    
class PhAINeuralNetwork1(nn.Module):
    """
    Change of PhAI from paper - input and output have different dimensionality.
    """
    def __init__(self, *, max_index, filters, kernel_size, cnn_depth, dim, dim_exp, dim_token_exp, mlp_depth, reflections, data_size_mode):
        super().__init__()
        self.data_size_mode = data_size_mode
        if data_size_mode == "full_8":
            hkl = [16, 16, 16]
        elif data_size_mode == "half_8":
            hkl = [16,16,8]
        elif data_size_mode == "full_16":
            hkl = [32,32,32]
        elif data_size_mode == "half_16":
            hkl = [32,32,16]
        mlp_token_nr  = filters
        padding       = int((kernel_size - 1) / 2)
        self.sigmoid = nn.Sigmoid()
        self.net_a = nn.Sequential(
            #Rearrange('b x y z  -> b 1 x y z '),

            nn.Conv3d(1, filters, kernel_size = kernel_size, padding=padding),
            nn.GELU(),
            nn.GroupNorm(filters, filters)
        )

        self.net_convolution_layers = nn.Sequential(
            *[nn.Sequential(
                ConvolutionalBlock(filters, kernel_size = kernel_size, padding = padding),
            ) for _ in range(cnn_depth)],
        )

        self.net_projection_layer = nn.Sequential(
            Rearrange('b c x y z  -> b c (x y z)'),
            #nn.Linear(hkl[0]*hkl[1]*hkl[2], dim),
        )

        self.net_mixer_layers = nn.Sequential(
            *[nn.Sequential(
                MLPLayer(mlp_token_nr, hkl[0]*hkl[1]*hkl[2], dim_token_exp, 'token'),
                MLPLayer(mlp_token_nr, hkl[0]*hkl[1]*hkl[2], dim_exp      , 'channel'),
            ) for _ in range(mlp_depth)],
            nn.LayerNorm(hkl[0]*hkl[1]*hkl[2]),
        )

        self.net_output = nn.Sequential(
            Rearrange('b t x -> b x t'),
            nn.AdaptiveAvgPool1d(1),
            Rearrange('b x 1 -> b x'),
            Rearrange('b x -> b 1 x'),
        )
        

    def forward(self, input_amplitudes):

        a = self.net_a(input_amplitudes)
        
        x = self.net_convolution_layers(a)

        x = self.net_projection_layer(x)

        x = self.net_mixer_layers(x)

        phases = self.net_output(x)
        #sigmoid_phases = self.sigmoid(phases)
        return phases


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout3d(0.3),  # PĹ™idĂˇn dropout pro regularizaci
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffZ = x2.size(2) - x1.size(2)
        diffY = x2.size(3) - x1.size(3)
        diffX = x2.size(4) - x1.size(4)
        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                                    diffY // 2, diffY - diffY // 2,
                                    diffZ // 2, diffZ - diffZ // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class PerSampleMinMaxNormalize:
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        min_val = tensor.min()
        max_val = tensor.max()
        return (tensor - min_val) / (max_val - min_val + 1e-8)


class UNet3D(nn.Module):
    """
    Standart UNet 3D with 4 downsample and 4 upsamle block
    """
    def __init__(self, n_channels, n_classes):
        super(UNet3D, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        
        self.inc = DoubleConv(n_channels, 128)
        self.down1 = Down(128, 256)
        self.down2 = Down(256, 512)
        self.down3 = Down(512, 1024)
        self.down4 = Down(1024, 2048)
        self.up1 = Up(2048, 1024)
        self.up2 = Up(1024, 512)
        self.up3 = Up(512, 256)
        self.up4 = Up(256, 128)
        self.outc = OutConv(128, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
