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
import shutil
from tqdm import tqdm
import csv

from cnn import CNNController
from data_generator import DataGenerator
from utils import *

# Main -- Cosine one BUT BINARIZED

def train(model, data_generator, optimizer, criterion, device, n_bits, var, D=512, n_step = 50000, save=True, sharpen='softmax'):
    
    exp_name = f"{W}way{S}shot{D}dim"
    writer = SummaryWriter(log_dir='./log', comment=exp_name)
  
    loss_train = []
    val_accs = []
    steps = []
    best_acc = 0
    for step in tqdm(range(n_step)):
        #prep data
        support_label, support_set, query_label, query_set = data_generator.sample_batch('train', 32)
        support_label, support_set = prep_data(support_label, support_set, device)
        query_label, query_set = prep_data(query_label, query_set, device)
        #support set loading
        support_keys = None
        model.eval()
        with torch.no_grad():
            support_keys = model(support_set) # output: d-dim real vector
        
        #query evaluation
        model.train()
        query_keys = model(query_set)
        cosine_sim = get_cosine_similarity(query_keys, support_keys)
        if sharpen == 'softmax':
            sharpened = sharpening_softmax(euc_dis)
        elif sharpen == 'softabs':
            sharpened = sharpening_softabs(euc_dis, 10)

        normalized = normalize(sharpened)
        pred = weighted_sum(normalized, support_label)    
        optimizer.zero_grad()
        loss = criterion(pred, query_label)
        loss.backward()
        if step % 500 == 0:
            print(f"train loss = {loss}")
            writer.add_scalar("Loss/Train", loss, step)
            acc = inference(model, data_gen, device, 3, 0.2, key_mem_transform=None, n_step=250, type='val')
            print(f"val acc = {acc}")
            writer.add_scalar("Acc/Val", acc, step)
            loss_train.append(loss.detach().cpu().numpy())
            val_accs.append(acc)
            steps.append(step)

            # save model
            if save:
                torch.save(model.state_dict(), f"./results/{exp_name}_checkpoint.pth.tar")
                if acc > best_acc:
                    best_acc = acc
                    shutil.copy(f"./results/{exp_name}_checkpoint.pth.tar", f"./results/{exp_name}_best.pth.tar")

        #backprop
        optimizer.step()
    return model, steps, loss_train, val_accs


def inference(model, data_generator, device, n_bits, data, key_mem_transform = binarize, n_step = 100, sum_argmax=True, type='val'):
  model.eval()
  accumulated_acc = 0
  if key_mem_transform in (bipolarize, binarize):
    for i in range(n_step):
      support_label, support_set, query_label, query_set = data_generator.sample_batch(type, 32)
      support_label, support_set = prep_data(support_label, support_set, device)
      query_label, query_set = prep_data(query_label, query_set, device)
      support_label = support_label.cpu().numpy()
      query_label = query_label.cpu().numpy()
      with torch.no_grad():
        support_keys = key_mem_transform(model(support_set).cpu().detach().numpy())
        query_keys = key_mem_transform(model(query_set).cpu().detach().numpy())
       
        ### CSV VARIATION (with new numpy function)
        support_keys_probVariation = csv_numpy_variation(support_keys, n_bits, data)
        # have to change the combining of them to support the numpy characteristics now too
        support_keys = np.concatenate((support_keys, support_keys_probVariation), axis=0)
        
        dot_sim = get_dot_prod_similarity(query_keys, support_keys)
        sharpened = np.abs(dot_sim)

        if sum_argmax:
          pred = np.dot(sharpened, support_label)
          pred_argmax = np.argmax(pred, axis=1)
        else:
          support_label_argmax = np.argmax(support_label, axis=1)
          pred_argmax = support_label_argmax[sharpened.argmax(axis=1)]
        
        query_label_argmax = np.argmax(query_label, axis = 1)
        accumulated_acc += np.sum(pred_argmax == query_label_argmax)/len(pred_argmax)
    return accumulated_acc/n_step
  else:
    for i in range(n_step):
      support_label, support_set, query_label, query_set = data_generator.sample_batch(type, 32)
      support_label, support_set = prep_data(support_label, support_set, device)
      query_label, query_set = prep_data(query_label, query_set, device)
      support_label = support_label
      query_label = query_label
      with torch.no_grad():
        ### LINEAR QUANTIZATION ###
        support_keys = quantize(model(support_set), n_bits)
        query_keys = quantize(model(query_set), n_bits)

        ### NORMAL VARIATION ###
        #support_keys_normalVariation = normal_dist_variation(support_keys, n_bits, var)
        #support_keys = support_keys + support_keys_normalVariation.to(device)
        
        ### PROBABILITY-BASED VARIATION ###
        #support_keys_probVariation = prob_dist_variation(support_keys, n_bits, 0.568)
        #support_keys = support_keys + support_keys_probVariation.to(device)
        
        ### CSV VARIATION ###
        support_keys_probVariation = csv_prob_dist_variation(support_keys, n_bits, data)
        support_keys = support_keys + support_keys_probVariation.to(device)

        ### COSINE SIM ###
        cosine_sim = get_cosine_similarity(query_keys, support_keys)
        sharpened = sharpening_softabs(cosine_sim, 10)
        normalized = normalize(sharpened)
        
        if sum_argmax:
          pred = weighted_sum(normalized, support_label).cpu().numpy()
          pred_argmax = np.argmax(pred, axis=1)
        else:        
          support_label_argmax = np.argmax(support_label.cpu().numpy(), axis=1)
          pred_argmax = support_label_argmax[normalized.cpu().numpy().argmax(axis=1)]  #Cosine
        query_label_argmax = np.argmax(query_label.cpu().numpy(), axis=1)
        accumulated_acc += np.sum(pred_argmax == query_label_argmax) / len(pred_argmax)
    return accumulated_acc / n_step


if __name__ == '__main__':
  device = torch.device('cpu')
  if torch.cuda.is_available():
    device = torch.device('cuda')

  print("COSINE MODEL W BINARIZE AND CVS VARIATION")

  for w, s in [[5, 1], [20, 5], [100, 5]]:             # [[5, 1], [20, 5], [100, 5]]
    for b in [1, 2, 3]:                   # [1, 2, 3]
      for d in [20, 30, 50, 100, 150]:                # [20, 30, 50, 100, 150]
        W = w       #way
        S = s       #shots
        D = 512     #dimensions
        dom = d     #domains
        n_bits = b  #bits
        exp_name = f"{W}way{S}shot{D}dim"

        data_gen = DataGenerator(W,S)
        model = CNNController(D).float().to(device)

        csv_file_path = f"./var_data/{n_bits}bit/{dom}dom.csv"
        with open(csv_file_path, 'r') as file:
            csv_reader = csv.reader(file)
            data = list(csv_reader)

        ### TRAINING ###
        #criterion = nn.CrossEntropyLoss()
        #criterion = nn.BCELoss()
        #optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-5)
        #model.load_state_dict(torch.load(f'./results/{D}dim/{exp_name}_best.pth.tar'))
        #model, steps, loss, acc = train(model, data_gen, optimizer, criterion, device, D, 50000, sharpen='softmax')
        #model, steps, loss, acc = train(model, data_gen, optimizer, criterion, device, 3, 0.2, D, 50000, sharpen='softmax')
        #steps = np.asarray(steps)
        #loss = np.asarray(loss)
        #acc = np.asarray(acc)

        ### EVALUATION/INFERENCE ###
        model.load_state_dict(torch.load(f'./results_cos/{exp_name}_best.pth.tar'))
        acc = inference(model, data_gen, device, n_bits, data, sum_argmax=False, type='test')
        print(f"{W}-way {S}-shot {D}-dim {n_bits}-bits {dom}-domains")
        print(f"acc = {acc}")
