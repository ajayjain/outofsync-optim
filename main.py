#!/usr/bin/env python3
# Trainer adapted from https://github.com/pytorch/examples

from __future__ import print_function
import argparse
import sys
import datetime 
import os
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import datasets, transforms
from torch.autograd import Variable

from tensorboardX import SummaryWriter

from utils import *
from Run import Run

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--delay', default=0, type=int,
                    help='the number of batches to buffer and replay - '
                    'i.e. the gradient delay')
parser.add_argument('--log-output', action='store_true', default=False,
                    help='logs output to tensorboard and CSV')
parser.add_argument('--optimizer', type=str, default="sgd",
                    help='select optimizer. Defaults to SGD')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--task', type=str, default='MNIST', 
                    help='which task to train, MNIST or CIFAR10')
parser.add_argument('--tensorboard-plot', type=str, default='',
                    help='name of the plot in which to store the results')
parser.add_argument('--warmup', type=str, default='none',
                    help='learning rate warmup method to use, gradual or constant')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

run = Run(
    task = args.task,
    learning_rate = args.lr,
    batch_size = args.batch_size,
    delay = args.delay,
    warmup = args.warmup,
    momentum = args.momentum,
    optimizer = args.optimizer
    )

run_name = run.to_filename()

writer = SummaryWriter('runs/' + run_name)

# Datasets
train_loader = torch.utils.data.DataLoader(
    get_training_set(args),
    batch_size=args.batch_size, shuffle=True, **kwargs)

test_loader = torch.utils.data.DataLoader(
    get_test_set(args),
    batch_size=args.test_batch_size, shuffle=True, **kwargs) 


# Logging

## Generate runs-csv folder if not existent
os.makedirs('runs-csv', exist_ok=True)

## Generate new log folder on each run of this file
if args.log_output:
    if os.path.exists('runs-csv/' + run_name):
        shutil.rmtree('runs-csv/' + run_name)
    os.makedirs('runs-csv/' + run_name)
    open('runs-csv/' + run_name + '/train_accuracy', 'w').close()
    open('runs-csv/' + run_name + '/train_loss', 'w').close()
    open('runs-csv/' + run_name + '/test_accuracy', 'w').close()
    open('runs-csv/' + run_name + '/test_loss', 'w').close()


def log_scalar(name, x, y):
    if args.log_output:
        writer.add_scalar(args.tensorboard_plot + '/' + name, y, x)
        with open('runs-csv/' + run_name + '/' + name, 'a') as f:
            f.write(str(x) + ',' + str(y) + '\n')

# Training & Testing
class DelayedOptimizer():
    def __init__(self, optimizer, delay):
        super(DelayedOptimizer, self).__init__()

        self.delay = delay
        self.optimizer = optimizer
        self.gradient_buf = []

    def step(self):
        if self.delay > 0:
            new_grad = get_gradients(self.optimizer)
            self.gradient_buf.append(new_grad)

            if len(self.gradient_buf) > self.delay:
                self._apply_oldest()
        else:
            self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def _apply_oldest(self):
        # Replay the oldest gradient and update parameters
        old_grad = self.gradient_buf.pop(0)
        set_gradients(self.optimizer, old_grad)
        self.optimizer.step()

    def apply_all(self):
        # Apply all updates from buffer
        while self.gradient_buf:
            self._apply_oldest()


model = get_model(args.task)
if args.cuda:
    model.cuda()
if args.optimizer == "adam":
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
else:
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
delayed_optimizer = DelayedOptimizer(optimizer, args.delay)

# Train & test

def train(epoch):
    model.train()

    # Optional: learning rate warmup
    if args.warmup == 'gradual':
        set_learning_rate(delayed_optimizer.optimizer, gradual_warmup(epoch, target_lr=args.lr))
    elif args.warmup == 'constant':
        set_learning_rate(delayed_optimizer.optimizer, constant_warmup(epoch, target_lr=args.lr))

    train_loss = 0
    train_correct = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()

        data, target = Variable(data), Variable(target)
        delayed_optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        delayed_optimizer.step()

        # Compute training accuracy & loss
        pred = output.data.max(1, keepdim=True)[1]
        train_correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        train_loss += loss.data[0]

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))



    # Log average train loss over the epoch
    train_loss /= len(train_loader.dataset)
    log_scalar('train_loss', epoch, train_loss)

    # Log average train accuracy over the epoch
    train_accuracy = 100. * train_correct / len(train_loader.dataset)
    log_scalar('train_accuracy', epoch, train_accuracy)

    print ('Train finished.')

def test(epoch):
    model.eval()
    test_loss = 0
    correct = 0
    for test_idx, (data, target) in enumerate(test_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        batch_loss = F.nll_loss(output, target, size_average=False).data[0]
        test_loss += batch_loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        batch_correct = pred.eq(target.data.view_as(pred)).cpu().sum()
        correct += batch_correct



    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        test_accuracy))

    log_scalar('test_loss', epoch, test_loss)

    log_scalar('test_accuracy', epoch, test_accuracy)


for epoch in range(1, args.epochs + 1):

    # Do one test without any training, to get a baseline.
    test(epoch)
    train(epoch)

# Do a final test after the last train.
test(args.epochs + 1)


writer.close()

