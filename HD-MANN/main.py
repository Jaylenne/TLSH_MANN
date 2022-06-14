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

from cnn import CNNController
from data_generator import DataGenerator
from utils import *

def train(model, data_generator, optimizer, criterion, device, D=512, n_step = 50000, save=True, sharpen='softmax'):
    
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
    # sharpened = sharpening_softabs(cosine_sim, 10)
    if sharpen == 'softmax':
      # print("Using softmax sharpening function")
      sharpened = sharpening_softmax(cosine_sim)
      # print(np.shape(sharpened))
    elif sharpen == 'softabs':
      # print('Using softabs sharpening function')
      sharpened = sharpening_softabs(cosine_sim, 10)
      # print(np.shape(sharpened))

    normalized = normalize(sharpened)
    pred = weighted_sum(normalized, support_label)
    optimizer.zero_grad()
    loss = criterion(pred, query_label)
    loss.backward()
    if step % 500 == 0:
      print(f"train loss = {loss}")
      writer.add_scalar("Loss/Train", loss, step)
      acc = inference(model, data_gen, device, key_mem_transform='full', n_step=250, type='val')
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


def inference(model, data_generator, device, key_mem_transform = binarize, n_step = 1000, sum_argmax=True, type='val'):
  model.eval()
  accumulated_acc = 0
  if key_mem_transform in (bipolarize, binarize):
    for i in tqdm(range(n_step)):
      support_label, support_set, query_label, query_set = data_generator.sample_batch(type, 32)
      support_label, support_set = prep_data(support_label, support_set, device)
      query_label, query_set = prep_data(query_label, query_set, device)
      support_label = support_label.cpu().numpy()
      query_label = query_label.cpu().numpy()
      with torch.no_grad():
        support_keys = key_mem_transform(model(support_set).cpu().detach().numpy())
        query_keys = key_mem_transform(model(query_set).cpu().detach().numpy())
        dot_sim = get_dot_prod_similarity(query_keys, support_keys)
        sharpened = np.abs(dot_sim)
        if sum_argmax:
          pred = np.dot(sharpened, support_label)
          pred_argmax = np.argmax(pred, axis=1)
        else:
          support_label_argmax = np.argmax(support_label, axis=1)
          pred_argmax = support_label_argmax[sharpened.argmax(axis=1)]
        query_label_argmax = np.argmax(query_label, axis = 1)
        # print(np.sum(pred_argmax == query_label_argmax))
        accumulated_acc += np.sum(pred_argmax == query_label_argmax)/len(pred_argmax)
    return accumulated_acc/n_step
  else:
    for i in tqdm(range(n_step)):
      support_label, support_set, query_label, query_set = data_generator.sample_batch(type, 32)
      support_label, support_set = prep_data(support_label, support_set, device)
      query_label, query_set = prep_data(query_label, query_set, device)
      support_label = support_label
      query_label = query_label
      with torch.no_grad():
        support_keys = model(support_set)
        query_keys = model(query_set)
        cosine_sim = get_cosine_similarity(query_keys, support_keys)
        sharpened = sharpening_softabs(cosine_sim, 10)
        normalized = normalize(sharpened)
        if sum_argmax:
          pred = weighted_sum(normalized, support_label).cpu().numpy()
          pred_argmax = np.argmax(pred, axis=1)
        else:
          support_label_argmax = np.argmax(support_label.cpu().numpy(), axis=1)
          pred_argmax = support_label_argmax[normalized.cpu().numpy().argmax(axis=1)]
        query_label_argmax = np.argmax(query_label.cpu().numpy(), axis=1)
        accumulated_acc += np.sum(pred_argmax == query_label_argmax) / len(pred_argmax)
    return accumulated_acc / n_step


if __name__ == '__main__':
  device = torch.device('cpu')
  if torch.cuda.is_available():
    device = torch.device('cuda')

  for dim in [1024, 2048]:
    W = 100 #way
    S = 5 #shots
    D = dim
    exp_name = f"{W}way{S}shot{D}dim"
    print(exp_name)

    data_gen = DataGenerator(W,S)
    model = CNNController(D).float().to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-4)

    model, steps, loss, acc = train(model, data_gen, optimizer, criterion, device, D, 50000, sharpen='softabs')

    steps = np.asarray(steps)
    loss = np.asarray(loss)
    acc = np.asarray(acc)

    # evaluation
    model.load_state_dict(torch.load(f'./results/{exp_name}_best.pth.tar'))
    acc = inference(model, data_gen, device, key_mem_transform=bipolarize, sum_argmax=False, type='test')
    print(f"acc = {acc}")

    # np.savez(f'{exp_name}.npz', steps, loss, acc)

