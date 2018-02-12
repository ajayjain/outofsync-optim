from torchvision import datasets, transforms

from models import *

def get_training_set(args):
    if args.task == 'MNIST':
        return datasets.MNIST('data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))
    elif args.task == 'CIFAR10': 
        return datasets.CIFAR10('data', train=True, download=True, 
                    transform=transforms.Compose([
                        transforms.RandomCrop(32, padding=4),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                    ]))

def get_test_set(args): 
    if args.task == 'MNIST':
        return datasets.MNIST('data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))
    elif args.task == 'CIFAR10':
        return datasets.CIFAR10('data', train=False, download=True, 
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                    ]))

def get_gradients(optimizer):
    gradient_map = {}

    for group in optimizer.param_groups:
        for p in group["params"]:
            if p.grad is not None:
                gradient_map[p] = p.grad.data.clone()

    return gradient_map


def set_gradients(optimizer, gradient_map):
    for group in optimizer.param_groups:
        for p in group["params"]:
            if p.grad is not None:
                p.grad.data.copy_(gradient_map[p])

                # Debug: set gradient to zero
                # p.grad.data.copy_(torch.zeros_like(p.grad.data))

def get_model(args):
    model = None
    if args.task == 'MNIST':
        model = BasicConv()
    elif args.task == 'CIFAR10':
        model = MobileNetV2()

    if args.cuda:
        model.cuda()
    return model 

def set_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def gradual_warmup(current_epoch, max_epochs, target_lr=0.1, start_lr=0.01):
    return ((target_lr - start_lr) / max_epochs) * current_epoch + start_lr

def constant_warmup(current_epoch, jump_after=4, target_lr=0.1, start_lr=0.01):
    if current_epoch >= jump_after:
        return target_lr
    else:
        return start_lr