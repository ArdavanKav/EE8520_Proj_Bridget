import numpy as np
import torch
import torchio as tio
import torch.nn.functional as F
import monai.transforms as montrans
import math

def rescale_intensity_3D(img):
    img = img[np.newaxis, :, :, :]
    img = tio.RescaleIntensity(out_min_max=(0, 1))(img)
    img = img.squeeze(0)
    return img


def CropOrPad_3D(img, resolution):
    img = img[np.newaxis, :, :, :]
    img = tio.CropOrPad(resolution)(img)
    img = img.squeeze(0)
    return img


def process_tabular_data(tabular_data, label=None):
    
    # handle the missing entries in the tabular data: append the missing indicator mask at the end of the tabular data
    # ADNI tabular_data_attr (13 features): ['AGE', 'GENDER', 'EDUCATION', 'CSFVol', 'TotalGrayVol', 'CorticalWhiteMatterVol',
    #     'Left-Hippocampus', 'Right-Hippocampus', 'rh_entorhinal_thickness', 'lh_entorhinal_thickness',
    #     'APOE4', 'MMSE', 'ADAS13']
    # MCSA tabular_data_attr (5 features): ['AGE', 'GENDER', 'EDUCATION', 'MMSE', 'APOE4']
    
    tabular_data = np.array(tabular_data).astype(np.float32)
    
    # Handle missing values and create mask
    tabular_mask = np.isnan(tabular_data)
    tabular_mask = np.logical_not(tabular_mask).astype(np.float32)
    tabular_data = np.nan_to_num(tabular_data, copy=False)
    
    # Concat the mask to the tabular data (doubles the dimension)
    tabular_data = np.concatenate((tabular_data, tabular_mask), axis=0)
  
    tabular_data = torch.from_numpy(tabular_data)
    return tabular_data

