import os
import sys
import numpy as np
import random
import logging
import datetime
import shutil
import os.path as osp
import argparse
import configargparse

import torch
import torchvision
import torchvision.transforms as transforms
# from torch.utils.tensorboard import SummaryWriter

from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import omniglot
import memory

from lib_simlsh import *
from simArrayPy import simArrayPy


global args

parser = configargparse.ArgParser()
parser.add('-c', '--config', required=False,
           is_config_file=True, help='config file')
parser.add_argument('--save-dir', default=None, type=str, help='Path to storing the model')
parser.add_argument('--eval-way', default=5, type=int, help='Evaluation number of ways')
parser.add_argument('--eval-shot', default=1, type=int, help='Evaluation number of shots')
parser.add_argument('--eval-episode', default=1000, type=int, help='Evaluation iterate episodes')
parser.add_argument('--memory_size', default=2048, type=int, help='Memory capacity')
parser.add_argument('--key-dim', default=64, type=int, help='Key dimension extracted')
parser.add_argument('--lshdim', type=int, nargs='+', default=[64, 128, 256, 512, 1024, 2048, 4096],
                    help='Hyperplane dimension')
parser.add_argument('--asize', default=64, type=int, help='Crossbar array size')
parser.add_argument('--seed', default=43, type=int, help='Random seed')
parser.add_argument('--ideallshsim', action='store_true', help='Do the ideal LSH simulation')
parser.add_argument('--crossbarsim', action='store_true', help='Do crossbar simulation')
parser.add_argument('--update', action='store_true', help='Do the binary memory update')
parser.add_argument('--sum-argmax', action='store_true', help='Do the sum argmax')
parser.add_argument('--real-eval', action='store_true', help='Do the real value model evaluation')
parser.add_argument('--ch-last', default=128, type=int,
                    help='Channel number of the last convolution layers in CNN, to match the parameter count')


class Net(nn.Module):
    def __init__(self, input_shape, keydim=128, ch_last=args.ch_last):
        super(Net, self).__init__()
        # Constants
        kernel = 3
        pad = int((kernel - 1) / 2.0)
        p = 0.3

        ch, row, col = input_shape
        self.conv1 = nn.Conv2d(ch, 64, kernel, padding=(0, 0))
        self.conv2 = nn.Conv2d(64, 64, kernel, padding=(0, 0))
        self.conv3 = nn.Conv2d(64, 128, kernel, padding=(pad, pad))
        self.conv4 = nn.Conv2d(128, 128, kernel, padding=(pad, pad))
        self.conv5 = nn.Conv2d(128, ch_last, kernel, padding=(pad, pad))
        self.conv6 = nn.Conv2d(ch_last, ch_last, kernel, padding=(pad, pad))
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(9 * ch_last, keydim)
        self.dropout = nn.Dropout(p)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.dropout(x)
        return x


def main():
    # Set up logging
    datestr = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    fh = logging.FileHandler('./LSHomni/log/' + datestr + '.log')
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)

    setup_seed(args.seed)

    # Dataset loading
    DATA_FILE_FORMAT = os.path.join(os.getcwd(), '%s_omni.pkl')

    test_filepath = DATA_FILE_FORMAT % 'test'
    testset = omniglot.OmniglotDataset(test_filepath)
    testloader = testset.test_sampler(args.eval_way, args.eval_shot, args.eval_episode)

    logger.info('Test Dataset loaded')

    # Network initializing
    net = Net(input_shape=(1, 28, 28), keydim=args.key_dim)
    mem = memory.Memory(args.memory_size, args.key_dim)
    net.add_module("memory", mem)
    net.cuda()

    model_checkpoint = torch.load(args.save_dir)
    net.load_state_dict(model_checkpoint['model_state_dict'])

    logger.info(f'Model parameters loaded Dir: {args.save_dir}')
    logger.info(f"{args.eval_way}-way {args.eval_shot}-shot evaluation")

    net.eval()
    Acc_all = []
    if args.real_eval:
        logger.info(f"Evaluate specific {args.eval_way}-way {args.eval_shot}-shot")
        evaluation_all = []
        for data in tqdm(testloader):
            support_x, support_y, query_x, query_y = data
            evaluation = eval_fewshot(net, mem, support_x, support_y, query_x, query_y)
            evaluation_all.extend(evaluation)

        logger.info(f"{args.eval_way}-way {args.eval_shot}-shot mean accuracy: {np.mean(evaluation_all)}")
        logger.info(f"{args.eval_way}-way {args.eval_shot}-shot maximum accuracy: {np.array(evaluation_all).reshape(-1, args.eval_episode).mean(axis=0).max()}")
        np.savez(f'./LSHomni/results/RealEval_{args.eval_way}way_{args.eval_shot}shot_{args.key_dim}dim'+datestr, Acc_all=evaluation_all)
    else:
        for i, data in tqdm(enumerate(testloader)):
            accs = []
            # Extracting feature vectors using model
            support_x, support_y, query_x, query_y = data
            query_x = query_x.cuda()
            support_x = torch.cat(support_x, dim=0).cuda()
            support_y = torch.tensor(support_y)

            support_x_embed = net.memory.extract(net(support_x))
            query_x_embed = net.memory.extract(net(query_x))
            # LSH+TCAM simulation
            # LSH using crossbar arrays
            train_input = support_x_embed.detach().cpu().numpy()
            test_input = query_x_embed.detach().cpu().numpy()
            train_label = support_y.detach().numpy()
            test_label = query_y.detach().numpy()
            for d in args.lshdim:
                if args.ideallshsim:
                    acc = metric_hamming(train_input, test_input, train_label, test_label, dim_plane=d, update=args.update, sum_argmax=args.sum_argmax)
                    accs.append(acc)
                if args.crossbarsim:
                    Gmap = np.exp(np.random.randn(args.key_dim, d + 1) * 1.1 + 0.8)
                    G = g_reconstruct(Gmap, r_size=args.asize, c_size=args.asize)

                    train_hashcode, test_hashcode = crossbarlsh_wr_app(train_input, test_input, train_label, test_label, G,
                                                                       hashbits=d, bias=0.4, method='TLSH')
                    # TCAM and accuracy calculations
                    if args.update:
                        train_hashcode_update, train_label_update = memoryupdate_binary(train_hashcode, train_label)
                        acc = crossbartcam_wr_app(train_hashcode_update, test_hashcode, train_label_update, test_label, size=args.asize)
                    else:
                        acc = crossbartcam_wr_app(train_hashcode, test_hashcode, train_label, test_label, size=args.asize)
                    accs.append(acc)

            Acc_all.append(accs)
            tem_acc = np.array(Acc_all)

            if i % 20 == 0:
                for ld in range(len(args.lshdim)):
                    logger.info(f"Average accuracy at {i+1} episode: {tem_acc.mean(axis=0)[ld]}")

        Acc_all = np.array(Acc_all)
        for j in range(len(args.lshdim)):
            logger.info(f"LSH dim: {args.lshdim[j]}, Mean accuracy over {args.eval_episode}: {Acc_all.mean(axis=0)[j]}")

        np.savez(f'./LSHomni/results/LSHomni_ideal_{args.eval_way}-way_{args.eval_shot}-shot_{args.eval_episode}-'
                 f'episode_lshdim-{args.lshdim}_arraysize-{args.asize}_keydim-{args.key_dim}_'+datestr, Acc_total=Acc_all)


def eval_fewshot(model, mem, support_x, support_y, query_x, query_y):
    """
    Perform one N-way K-shot evaluation
    Return:
    """
    model.eval()
    mem.build()  # clear the memory
    # Update the memory for N-way K-shot images
    for xx, yy in zip(support_x, support_y):
        xx_cuda, yy_cuda = xx.cuda(), yy.cuda()
        query = model(xx_cuda)
        mem.query(query, yy_cuda, True)

    # Use remaining images to do evaluation on the updated memory
    query_x_cuda = query_x.cuda()
    query = model(query_x_cuda)
    yy_hat, _ = mem.predict(query)
    evaluation = torch.eq(yy_hat.detach().cpu(), query_y.unsqueeze(dim=1)).squeeze().numpy().astype('float')
    return evaluation


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True


def g_reconstruct(g, r_size=128, c_size=128, **kwargs):
    """
    input: Weight conductance before adding wire resistance /(uS)
    output: Effecitive conductance matrix after adding wire resistance
    """
    rw = kwargs['rw'] if 'rw' in kwargs.keys() else 1e-6

    n_r = np.ceil(g.shape[0]/r_size).astype(int)
    n_c = np.ceil(g.shape[1]/c_size).astype(int)
    g_new = np.zeros((r_size * n_r, c_size * n_c))
    g_new[:g.shape[0], :g.shape[1]] = g
    g_eff = np.zeros((n_r*r_size, g.shape[1]))
    c_last = g.shape[1] - (n_c - 1) * c_size
    for i, g_i in enumerate(np.hsplit(g_new, n_c)):
        if i == n_c - 1:
            g_i = g_i[:, :c_last]
        for j, g_j in enumerate(np.vsplit(g_i, n_r)):
            g_eff[j*r_size:(j+1)*r_size, i*c_size:i*c_size +
                  g_j.shape[1]] = simArrayPy(g_j, rw).geff

    return g_eff[:g.shape[0], :]


def crossbarlsh_wr_app(lsh_mem, lsh_query, lsh_memkey, lsh_querykey, G, hashbits=128, slope=0.782, intercept=-2.168,
                       scale=1., bias=1., method='ACM'):
    # crossbar lsh simulation
    """
    slope, intercept, scale: parameters determined conductance fluctuation
    G : geff after including wire resistance
    """
    Accuracy = []
    # G = g_reconstruct(G, r_size=size, c_size=size)
    Acc = []
    mem = lsh_mem
    query = lsh_query
    memory_lsh = np.zeros((mem.shape[0], hashbits))
    query_lsh = np.zeros((query.shape[0], hashbits))
    memkey = lsh_memkey
    querykey = lsh_querykey
    for j, mm in enumerate(mem):
        mm_neg, mm_pos = vec_pn(mm)
        g = np.abs(Gdrift(G, slope, intercept, scale))
        I_pos = np.dot(mm_pos, g) * 0.2
        g = np.abs(Gdrift(G, slope, intercept, scale))
        I_neg = np.dot(mm_neg, g) * 0.2
        I = I_pos - I_neg
        I = np.squeeze(I)
        if method == 'ACM':
            memory_lsh[j] = ((I[:-1] - I[1:]) > 0).astype(int)
        elif method == 'TLSH':
            memory_lsh[j] = tlsh((I[:-1] - I[1:]), bias)
    for j, qq in enumerate(query):
        qq_neg, qq_pos = vec_pn(qq)
        g = np.abs(Gdrift(G, slope, intercept, scale))
        I_pos = np.dot(qq_pos, g) * 0.2
        g = np.abs(Gdrift(G, slope, intercept, scale))
        I_neg = np.dot(qq_neg, g) * 0.2
        I = I_pos - I_neg
        I = np.squeeze(I)
        if method == 'ACM':
            query_lsh[j] = ((I[:-1] - I[1:]) > 0).astype(int)
        elif method == 'TLSH':
            query_lsh[j] = tlsh((I[:-1] - I[1:]), bias)

    return memory_lsh, query_lsh


def crossbartcam_wr_app(m_lsh, q_lsh, lsh_memkey, lsh_querykey, std=5, size=64):
    # crossbar 2dpe-TCAM
    Accuracy = []
    mem_lsh = m_lsh
    query_lsh = q_lsh
    m_key = lsh_memkey
    q_key = lsh_querykey
    tcam_stor = tcam_storage(mem_lsh, 150, 0, method='2dpe')
    tcam_stor = programerr(tcam_stor, std)
    tcam_stor = g_reconstruct(tcam_stor, r_size=size, c_size=size)
    if tcam_stor.shape[0] < 64:
        tcam_stor_temp = np.zeros((64, tcam_stor.shape[1]))
        tcam_stor_temp[:tcam_stor.shape[0]] = tcam_stor
        tcam_stor = tcam_stor_temp
    if 2 * query_lsh.shape[1] < 64:
        search_input = np.zeros((64, query_lsh.shape[0]))
    else:
        search_input = np.zeros((2 * query_lsh.shape[1], query_lsh.shape[0]))
    for i in range(query_lsh.shape[0]):
        search_input[:, i] = tcam_input(
            query_lsh[i], 1, 0, '2dpe').reshape(-1)

    Accuracy.append(np.sum(
        (m_key[np.argmin((search_input.T * 0.2) @ tcam_stor, axis=1)] == q_key).astype(int)) / query_lsh.shape[0])

    return Accuracy


def tcam_logicalxor(vec1, vec2):
    a = np.zeros_like(vec1)
    b = abs(vec1 - vec2)

    idx = b > 1
    b[idx] = 0

    a = b
    return a


def memoryupdate_binary(mem_lsh, memkey):
    """
    core of memory update scheme
    """

    # counter
    def numcount(vec):
        a = np.zeros(vec.shape)
        for i, v in enumerate(vec):
            if v == 1:
                a[i] = 1

            elif v == 0:
                a[i] = -1

        return a

    # update

    def denumcount(vec):
        a = np.zeros(vec.shape)
        for i, v in enumerate(vec):
            if v > 0:
                a[i] = 1
            elif v < 0:
                a[i] = 0
            elif v == 0:
                a[i] = 3

        return a

    memory = []
    key = []
    count = []
    for i in range(mem_lsh.shape[0] // 5):
        memory.append(mem_lsh[i * 5])
        key.append(memkey[i * 5])
        count.append(numcount(mem_lsh[i * 5]))

    for i in range(4):
        Query = []
        query_key = []
        for j in range(mem_lsh.shape[0] // 5):
            Query.append(mem_lsh[i + 1 + j * 5])
            query_key.append(memkey[i + 1 + j * 5])
        for k, v in enumerate(Query):
            dist = []
            for l, m in enumerate(memory):
                dist.append(np.sum(tcam_logicalxor(v, m)))
            if key[np.argmin(dist)] == query_key[k]:
                count[np.argmin(dist)] += numcount(v)
                memory[np.argmin(dist)] = denumcount(count[np.argmin(dist)])
            else:
                memory.append(v)
                key.append(query_key[k])
                count.append(numcount(v))

    return np.array(memory), np.array(key)


def metric_hamming(train_input, test_input, train_label, test_label, dim_plane=1024, update=True, sum_argmax=False):
    h_plane = np.random.randn(train_input.shape[-1], dim_plane)
    train_hashcode = ((train_input @ h_plane) > 0).astype(int)

    if update:
        train_hashcode, train_label = memoryupdate_binary(train_hashcode, train_label)
        print(train_label.shape)

    test_hashcode = ((test_input @ h_plane) > 0).astype(int)
    hamming_d = tcam_logicalxor(test_hashcode[:, None, :], train_hashcode[None, :, :]).sum(-1)

    if sum_argmax:
        id = np.identity(train_label.max() + 1)
        onehot = []
        for i in train_label:
            onehot.append(id[i])
        onehot = np.vstack(onehot)
        hd_sum = hamming_d @ onehot
        idx = np.argmin(hd_sum, axis=-1)
        acc = (idx == test_label).astype(int).sum() / len(test_label)
    else:
        idx = np.array(train_label)[np.argmin(hamming_d, axis=-1)]
        acc = (idx == test_label).astype(int).sum() / len(test_label)

    return acc


if __name__ == '__main__':
    args = parser.parse_args('-c ./LSHomni/lshsim.config')

    main()