#!/usr/bin/env python3
# Trainer adapted from https://github.com/pytorch/examples

from __future__ import print_function
import argparse
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import datasets, transforms
from torch.autograd import Variable

from tensorboardX import SummaryWriter

from utils import *

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
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--task', type=str, default='MNIST', 
                    help='which task to train, MNIST or CIFAR10')
parser.add_argument('--run-name', type=str, default='', 
                    help='name of the run for tensorboard logging')
parser.add_argument('--plot-name', type=str, default='',
                    help='name of the plot in which to store the results')
parser.add_argument('--warmup', type=str, default='none',
                    help='learning rate warmup method to use, gradual or constant')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

writer = SummaryWriter() if args.run_name is '' else SummaryWriter('runs/' + args.run_name)

plot_name = args.task + '_' + args.plot_name

# Datasets
train_loader = torch.utils.data.DataLoader(
    get_training_set(args),
    batch_size=args.batch_size, shuffle=True, **kwargs)

test_loader = torch.utils.data.DataLoader(
    get_test_set(args),
    batch_size=args.test_batch_size, shuffle=True, **kwargs) ## TODO: what exactly is this shuffling doing? 


# Logging

def log_scalar(name, y, x):
    if args.log_output:
        writer.add_scalar(name, y, x)

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

        # Compute training accuracy
        pred = output.data.max(1, keepdim=True)[1]
        train_correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))

        # Log train loss per-batch
        log_scalar('data/' + plot_name + '/training_loss', 
                            loss.data[0], 
                            epoch * len(train_loader) + batch_idx
                            )

    # Log average train accuracy over the epoch
    train_accuracy = 100. * train_correct / len(train_loader.dataset)
    log_scalar('data/' + plot_name + '/training_accuracy', train_accuracy, epoch)

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

    log_scalar('data/' + plot_name + '/test_loss', test_loss, epoch)

    log_scalar('data/' + plot_name + '/test_accuracy', test_accuracy, epoch)

    return test_accuracy


for epoch in range(1, args.epochs + 1):

    # Do one test without any training, to get a baseline.
    test(epoch)
    train(epoch)

# Do a final test after the last train.
test(epoch)


writer.close()

