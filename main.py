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
parser.add_argument('--sync', action='store_true', default=False,
                    help='disables delayed gradient application')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--task', type=str, default='MNIST', 
                    help='which task to train, MNIST or CIFAR10')
parser.add_argument('--run-name', type=str, default='', 
                    help='name of the run for tensorboard logging')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

writer = SummaryWriter()

run_name = args.run_name + '(' + args.task + ')'

# Datasets

train_loader = torch.utils.data.DataLoader(
    get_training_set(args),
    batch_size=args.batch_size, shuffle=True, **kwargs)

test_loader = torch.utils.data.DataLoader(
    get_test_set(args),
    batch_size=args.test_batch_size, shuffle=True, **kwargs)


# Training & Testing

model = get_model(args)

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)


def train(epoch):
    model.train()

    gradient_buf = None

    for batch_idx, (data, target) in enumerate(train_loader):

        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()

        if args.sync:
            optimizer.step()
        else:
            if batch_idx == 0:
                # First batch, initialize buffer
                gradient_buf = get_gradients(optimizer)
            else:
                # Save and swap model gradients with buffer, and step
                new_buf = get_gradients(optimizer)
                set_gradients(optimizer, gradient_buf)
                gradient_buf = new_buf

                optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))

        writer.add_scalar('data/' + run_name + '/training_loss', 
                            loss.data[0], 
                            epoch * len(train_loader) + batch_idx
                            )

    # Update parameters with gradients from the last batch
    if not args.sync and gradient_buf:
        set_gradients(optimizer, gradient_buf)
        optimizer.step()

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
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        writer.add_scalar('data/' + run_name + '/test_loss', 
                            batch_loss,
                            epoch * len(test_loader) + test_idx
                            )

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


for epoch in range(args.epochs):
    train(epoch)
    test(epoch)

# export scalar data to JSON for external processing
writer.export_scalars_to_json("./all_scalars.json")
writer.close()
