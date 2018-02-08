import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

def init_param(model, mode='xavier_normal'):
    for m in model.modules():
        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
            if mode == 'xavier_normal':
                init.xavier_normal(m.weight)
            elif mode == 'kaiming_normal':
                init.kaiming_normal(m.weight)
            if m.bias is not None:
                init.constant(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            if m.weight is not None:
                init.constant(m.weight, 1)
                init.constant(m.bias, 0)

class AE1(nn.Module):
    def __init__(self):
        super(AE1, self).__init__()
        self.fc1 = nn.Linear(784, 400)
        self.fc2 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)
        init_param(self, mode='xavier_normal')

    def encode(self, x):
        x = x.view(-1, 784)
        return self.fc2(F.elu(self.fc1(x)))

    def decode(self, z):
        return F.sigmoid(self.fc4(F.elu(self.fc3(z))))

    def forward(self, x):
        embeds = self.encode(x)
        logits = self.decode(embeds)
        return logits, embeds

class VAE1(nn.Module):
    def __init__(self):
        super(VAE1, self).__init__()
        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)
        init_param(self, mode='xavier_normal')

    def encode(self, x):
        x = x.view(-1, 784)
        h1 = F.elu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            nrm = torch.autograd.Variable(std.data.new(std.size()).normal_())
            return nrm.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        return F.sigmoid(self.fc4(F.elu(self.fc3(z))))

    def forward(self, x):
        mu, logvar = self.encode(x)
        embeds = self.reparameterize(mu, logvar)
        logits = self.decode(embeds)
        return logits, embeds, mu, logvar

if __name__ == '__main__':
     model = AE1()
     model = VAE1()
