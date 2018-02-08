from __future__ import print_function
import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as DD
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
from torchvision.utils import save_image

from models import *

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                    help='input batch size for training (default: 100)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='momentum (default: 0.9)')
parser.add_argument('--beta', type=float, default=0.999,
                    help='beta, if adam is used (default: 0.999)')
parser.add_argument('--WD', type=float, default=0,
                    help='weight decay (default: 0)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--saveDir', default='results/AE/', type=str, metavar='DIR',
                    help='where to store result images')
args = parser.parse_args()

# Set random seed
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

# Create saveDir
try:
    os.makedirs(args.saveDir)
except FileExistsError:
    pass

# Prepare dataset and loader
kwargs = {'num_workers': 1, 'pin_memory': True}
data_root = 'data/mnist/'
transform_train = transforms.Compose([transforms.ToTensor()])
transform_test = transforms.Compose([transforms.ToTensor()])
train_set = datasets.MNIST(root=data_root, train=True, transform=transform_train, download=True)
test_set = datasets.MNIST(root=data_root, train=False, transform=transform_test, download=True)
train_loader = DD.DataLoader(train_set, batch_size=args.batch_size, sampler=DD.sampler.RandomSampler(train_set), **kwargs)
test_loader = DD.DataLoader(test_set, batch_size=args.batch_size, sampler=DD.sampler.SequentialSampler(test_set), **kwargs)

# Prepare model
model = AE1().cuda()

#optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.WD)
optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(args.momentum, args.beta), weight_decay=args.WD)
#scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.33)

def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, batch_data in enumerate(train_loader):
        # batch_data = next(iter(train_loader))
        cuvar = lambda x: Variable(x.cuda())
        data, _ = map(cuvar, batch_data)
        optimizer.zero_grad()
        logits, embeds = model(data)
        loss_recon = F.binary_cross_entropy(logits, data.view(-1, 784), size_average=False)
        loss = loss_recon
        train_loss += loss.data[0]
        loss.backward()
        optimizer.step()
    train_loss /= len(train_loader.dataset)
    print('====> Epoch: {} Average train loss: {:.4f}'.format(epoch, train_loss))

def test(epoch):
    model.eval()
    test_loss = 0
    for batch_idx, batch_data in enumerate(test_loader):
        cuvar = lambda x: Variable(x.cuda(), volatile=True)
        data, _ = map(cuvar, batch_data)
        logits, _ = model(data)
        test_loss += F.binary_cross_entropy(logits, data.view(-1, 784), size_average=False).data[0]
        if batch_idx == 0:
            n = min(data.size(0), 8)
            comparison = torch.cat([data[:n], logits.view(-1, 1, 28, 28)[:n]])
            save_image(comparison.data.cpu(), '{}/recon_{}.png'.format(args.saveDir, epoch), nrow=n)
    test_loss /= len(test_loader.dataset)
    print('====> Epoch: {} Average test loss: {:.4f}'.format(epoch, test_loss))

sample_embeds = Variable(torch.randn(64, 20).cuda(), volatile=True)
for epoch in range(1, args.epochs + 1):
    #scheduler.step()
    train(epoch)
    test(epoch)
    sample_images = model.decode(sample_embeds)
    save_image(sample_images.cpu().data.view(64, 1, 28, 28), '{}/sample_{}.png'.format(args.saveDir, epoch))
