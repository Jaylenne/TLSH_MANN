import os
import sys
import numpy as np
import random
import logging
import datetime
import shutil
import os.path as osp
import configargparse

import torch

import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import omniglot
import memory


parser = configargparse.ArgParser()
parser.add('-c', '--config', required=False,
           is_config_file=True, help='config file')
parser.add_argument('--seed', default=43, type=int, help='Random Seed')
parser.add_argument('--memory-size', default=2048, type=int, help='Memory size')
parser.add_argument('--key-dim', default=128, type=int, help='Key dimension')
parser.add_argument('--batch-size', default=16, type=int, help='Training episode batch size')
parser.add_argument('--episode-length', default=30, type=int, help='Episode length')
parser.add_argument('--episode-width', default=5, type=int, help='Number of distinct class in one episode')
parser.add_argument('--val-shot', default=5, type=int, help='Validation shot')
parser.add_argument('--val-way', default=5, type=int, help='Validation way')
parser.add_argument('--validation-frequency', default=50, help='Every so often validate the model')
parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate for Adam')
parser.add_argument('--eps', default=1e-4, type=float, help='Eps for Adam')
parser.add_argument('--margin', default=0.1, type=float, help='Triplet loss margin')
parser.add_argument('--train-model', action='store_true', help='Train the model')
parser.add_argument('--load-model', action='store_true', help='Load the previous model')
parser.add_argument('--save-model', action='store_true', help='Save the model')
parser.add_argument('--do-eval', action='store_true', help='Evaluate the model by N-way K-shot')
parser.add_argument('--eval-way', default=5, type=int, help='Evaluation way')
parser.add_argument('--eval-shot', default=1, type=int, help='Evaluation shot')
parser.add_argument('--eval-episode', default=1000, type=int, help='Evaluation episode')
parser.add_argument('--savedir', default=None, type=str, help='Model saving directory')

args = parser.parse_args()


class Net(nn.Module):
    def __init__(self, input_shape, keydim=128):
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
        self.conv5 = nn.Conv2d(128, 256, kernel, padding=(pad, pad))
        self.conv6 = nn.Conv2d(256, 256, kernel, padding=(pad, pad))
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(2304, keydim)
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


def weight_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True


def save_checkpoint(state, is_best, folder, filename='model_best.pth.tar'):
    if not osp.exists(folder):
        os.umask(0)
        os.makedirs(folder, mode=0o777, exist_ok=False)
    torch.save(state, folder + '/' + filename)
    if is_best:
        shutil.copyfile(folder + '/' + filename, folder + '/' + 'model_best.pth.tar')


def load_checkpoint(folder, is_best=True):
    filename = 'model_best.pth.tar' if is_best else 'model_best.pth.tar'
    path = osp.join(folder, filename)
    loaded_checkpoint = torch.load(path, map_location='cuda')
    return loaded_checkpoint


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


# Set up logging
datestr = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)
fh = logging.FileHandler('log/' + datestr + '.log')
fh.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)
console = logging.StreamHandler()
console.setFormatter(formatter)
logger.addHandler(console)

# Training parameters

setup_seed(args.seed)

logger.info(f'Memory size: {args.memory_size}')
logger.info(f'Batch size: {args.batch_size}')
logger.info(f'Key dimension: {args.key_dim}')
logger.info(f'Training episode length: {args.episode_length}')
logger.info(f'Training episode width: {args.episode_width}')
logger.info(f'Validation frequency: {args.validation_frequency}')
logger.info(f'Test way: {args.test_way}')
logger.info(f'Test shot: {args.test_shot}')
logger.info(f'Learning rate: {args.lr}')
logger.info(f'Eps for Adam: {args.eps}')
logger.info(f'Seed: {args.seed}')
logger.info(f'Triplet loss margin: {args.margin}')

# Dataset loading
DATA_FILE_FORMAT = os.path.join(os.getcwd(), '%s_omni.pkl')

train_filepath = DATA_FILE_FORMAT % 'train'
trainset = omniglot.OmniglotDataset(train_filepath)
trainloader = trainset.sample_episode_batch(args.episode_length, args.episode_width, args.batch_size, N=10000)

test_filepath = DATA_FILE_FORMAT % 'test'
testset = omniglot.OmniglotDataset(test_filepath)

logger.info('Dataset loaded')

# Network initializing
net = Net(input_shape=(1, 28, 28), keydim=args.key_dim)
mem = memory.Memory(args.memory_size, args.key_dim, margin=args.margin)
net.add_module("memory", mem)
net.cuda()
net.apply(weight_init)

optimizer = optim.Adam(net.parameters(), lr=args.lr, eps=args.eps)
lrscheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=20, verbose=True)

cummulative_loss = 0
counter = 0
best_val_acc = 0
episode_start = 0

if args.load_model:
    checkpoint_pre = load_checkpoint(args.savedir, True)
    net.load_state_dict(checkpoint_pre['model_state_dict'])
    optimizer.load_state_dict(checkpoint_pre['optimizer_state_dict'])
    lrscheduler.load_state_dict(checkpoint_pre['scheduler_state_dict'])
    episode_start = checkpoint_pre['episode']
    best_val_acc = checkpoint_pre['best_val_acc']
    logger.info('Load previous model')

if args.train_model:
    logger.info('Start Training')
    for i, data in tqdm(enumerate(trainloader, episode_start)):
        # erase memory before training episode
        net.train()
        mem.build()
        x, y = data
        is_best = False
        for xx, yy in zip(x, y):
            optimizer.zero_grad()
            xx_cuda, yy_cuda = xx.cuda(), yy.cuda()
            embed = net(xx_cuda)
            yy_hat, softmax_embed, loss = mem.query(embed, yy_cuda, False)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), max_norm=5.0)
            optimizer.step()
            cummulative_loss += loss.detach()  # loss across the whole (episode * val_frequency)
            counter += 1

        with torch.no_grad():
            if i % args.validation_frequency == 0:
                # validation
                correct = []
                correct_by_k_shot = dict((k, list()) for k in range(args.val_shot + 1))
                testloader = testset.sample_episode_batch((args.val_shot + 1) * args.val_way, args.val_way, batch_size=1, N=100)

                net.eval()
                for data in testloader:
                    # erase memory before validation episode
                    mem.build()

                    x, y = data
                    y_hat = []
                    for xx, yy in zip(x, y):
                        xx_cuda, yy_cuda = xx.cuda(), yy.cuda()
                        query = net(xx_cuda)
                        yy_hat, embed, loss = mem.query(query, yy_cuda, True)
                        y_hat.append(yy_hat)
                        correct.append(float(torch.equal(yy_hat.cpu(), torch.unsqueeze(yy, dim=1))))

                    # compute per_shot accuracies
                    seen_count = [0 for idx in range(args.val_way)]
                    # loop over episode steps
                    for yy, yy_hat in zip(y, y_hat):
                        count = seen_count[yy[0] % args.val_way]
                        if count < (args.val_shot + 1):
                            correct_by_k_shot[count].append(float(torch.equal(yy_hat.cpu(), torch.unsqueeze(yy, dim=1))))
                        seen_count[yy[0] % args.val_way] += 1

                temp_acc = np.mean(correct)
                if temp_acc > best_val_acc:
                    is_best = True
                    best_val_acc = temp_acc

                logger.info("episode batch: {0:d} average loss: {1:.6f}".format(i, (cummulative_loss / counter)))
                logger.info("validation overall accuracy {0:f}".format(temp_acc))
                for idx in range(args.val_shot + 1):
                    logger.info("{0:d}-shot: {1:.3f}".format(idx, np.mean(correct_by_k_shot[idx])))
                cummulative_loss = 0
                counter = 0

                lrscheduler.step(temp_acc)  # ReduceOnPlateu scheduler

            if args.save_model:
                checkpoint = {
                    'episode': i,
                    'best_val_acc': best_val_acc,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': lrscheduler.state_dict()
                }
                save_checkpoint(checkpoint, is_best, args.savedir)


if args.do_eval:
    logger.info(f"Evaluate specific {args.eval_way}-way {args.eval_shot}-shot")
    evalloader = testset.test_sampler(args.eval_way, args.eval_shot, args.eval_episode)
    evaluation_all = []
    for data in tqdm(evalloader):
        support_x, support_y, query_x, query_y = data
        evaluation = eval_fewshot(net, mem, support_x, support_y, query_x, query_y)
        evaluation_all.extend(evaluation)

    logger.info(f"{args.eval_way}-way {args.eval_shot}-shot: {np.mean(evaluation_all)}")
