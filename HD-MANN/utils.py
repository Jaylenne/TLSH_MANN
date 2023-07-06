import torch
import torch.nn as nn
import glob
import os
import sys
import numpy as np 
import random
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from more_itertools import unzip

def get_cosine_similarity(a, b):
  a_norm = a / a.norm(dim=1)[:, None]
  b_norm = b / b.norm(dim=1)[:, None]
  res = torch.mm(a_norm, b_norm.transpose(0,1))
  
  return res

def get_Euclidean_dist(a, b):
  res = torch.cdist(a, b, p=2)
  return res

def get_Euclidean_similarity(a, b):
  res = 1/1+torch.cdist(a, b, p=2)
  return res

def get_dot_prod_similarity(a,b):
  return np.dot(a,b.T)

def sharpening_softabs(x,beta=10):
  return 1/(1+torch.exp(-beta*(x-0.5)))+1/(1+torch.exp(-beta*(-x-0.5)))

def sharpening_softmax(x):
  res = torch.nn.functional.softmax(x)
  return res

def normalize(x):
  return x/torch.sum(x,axis=1).reshape(-1,1)

def weighted_sum(attention_vec, value_mem):
  return torch.mm(attention_vec,value_mem)

def binarize(keys):
  return np.where(keys>0, 1, 0)

def bipolarize(keys):
  return np.where(keys>0, 1, -1)

def image_file_to_array(filename, transform):
  image = Image.open(filename)
  image = transform(image)
  arr = np.asarray(image)
  return 1.0 - arr.astype(np.float32)

def prep_data(batch_labels, batch_imgs, device):
  B,H,W = batch_imgs.shape
  batch_imgs = batch_imgs.reshape((B,1,H,W))
  inputs = torch.tensor(batch_imgs).float().to(device)
  targets = torch.tensor(batch_labels).float().to(device)
  return targets, inputs

def quantize(x, num_bits=0):
    if num_bits == 0:
        q_x = x
    else:
        q_min_val = 0
        q_max_val = 2**num_bits - 1
        
        x_min_val = torch.min(x)
        x_max_val = torch.max(x)
        
        step_size = (x_max_val - x_min_val) / q_max_val
        q_x = torch.round((x - x_min_val) / step_size)
        q_x = q_x.clamp(q_min_val, q_max_val)

    return q_x

def normal_dist_variation(x, n_bits, var=0):
    scale = var * 2**n_bits
    normal_dist = torch.randn(x.shape[0], x.shape[1]) * scale
    return normal_dist

def prob_dist_variation(x, n_bits, var=0, p=0):
    num_elements = x.shape[0] * x.shape[1]
    num_varied_elements = int(num_elements * p)
    scale = var * 2**n_bits

    prob_dist = torch.zeros_like(x)
    indices = random.sample(range(num_elements), num_varied_elements)
 
    for idx in indices:
        row_idx = idx // x.shape[1]
        col_idx = idx % x.shape[1]
        prob_dist[row_idx, col_idx] = random.choice([-1, 1]) * scale
     
    return prob_dist

def csv_prob_dist_variation(x, n_bits, data):
    prob_dist = torch.zeros_like(x)
    num_cols = len(data[0])
    num_elements = x.shape[0] * x.shape[1]
    
    for idx in range(num_elements):
        support_key = int(x.flatten()[idx])
        col_idx = random.choice(range(0, num_cols))
        var = float(data[support_key][col_idx])

        dist_row_idx = idx // x.shape[1]
        dist_col_idx = idx % x.shape[1]
        prob_dist[dist_row_idx][dist_col_idx] = var * 2**n_bits

    return prob_dist
