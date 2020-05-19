import torch
import torch.nn as nn
from math import pi
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
import pyramid

import torchvision.models.resnet as res

def _resnet(arch, block, layers, pretrained, progress, **kwargs):

    model = ReSpaceNet(block, layers, **kwargs)
    if pretrained:
        state_dict = res.load_state_dict_from_url(res.model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


class ReSpaceNet(res.ResNet):
    def __init__(self, block, layers,**kwargs):
        super().__init__(block,layers,**kwargs)



    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        return x

    def patch(self):
        cw = self.conv1.weight

        self.conv1.stride = (1,1)
        self.conv1.weight = nn.Parameter(cw[:,[1],:,:].cuda())

        self.layer4 = None
        self.layer3[5].conv2.bias = nn.Parameter(torch.zeros(256))
        self.layer3[5].bn2 = nn.Sequential()

        self.fc = None

def mass2d(x):
    N = x.shape[0]
    C = x.shape[1]
    H = x.shape[2]
    W = x.shape[3]
    p = F.softmax(x.view(N,C,H*W), 2).view(N,C,H,W,1)

    h_range = torch.arange((H - 1) / H, -1, -2 / H, device=x.device)
    w_range = torch.arange(-(W - 1) / W, 1, 2 / W, device=x.device)
    h_grid,w_grid = torch.meshgrid(h_range,w_range)

    grid = torch.stack((h_grid, w_grid), 2).expand(N, C, H, W, 2)

    mass = grid*p

    mass = torch.cat((p*x.unsqueeze(4),mass),dim=4)

    com = mass.sum(dim=(2,3))
    return com, p.detach()


class Regressor(nn.Module):
    def __init__(self,levels):
        super().__init__()
        self.fc1 = nn.Linear(256*3 * levels, 512)
        nn.init.zeros_(self.fc1.bias)
        nn.init.orthogonal_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)

        self.fc2 = nn.Linear(512, 128)
        nn.init.orthogonal_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

        self.regressor = nn.Linear(128, 2)
        nn.init.orthogonal_(self.regressor.weight)
        nn.init.zeros_(self.regressor.bias)

    def forward(self,x):
        out = self.fc1(x)
        out = F.relu(out)

        out = self.fc2(out)
        out = F.relu(out)

        out = self.regressor(out)
        return out

class PyramidAttention(nn.Module):
    def __init__(self,levels):

        super().__init__()
        self.levels = levels

        self.resnet = _resnet('resnet34', res.BasicBlock,
                              # [3, 4, 23, 3],
                              [3, 4, 6, 3],
                              #[2,2,2,2],
                              True, True) #[3, 4, 23, 3]
        self.resnet.eval()

        self.resnet.patch()

        self.regressor = Regressor(levels)


    def forward(self, x,pos,train=False):
        multi = pos.shape[1]
        device = x[0].device
        theta = (torch.rand((pos.shape[0], pos.shape[1], 1, 1), device=device) * 2 - 1) * pi / 12
        scale = torch.exp((torch.rand((pos.shape[0], pos.shape[1], 1, 1), device=device) * 2 - 1)*0.05)
        if not train:
            theta = theta*0
            scale = torch.ones((pos.shape[0], pos.shape[1], 1, 1), device=device)
        rsin = theta.sin()
        rcos = theta.cos()

        H = x[0].shape[2]
        W = x[0].shape[3]

        pos_fix = pos.clone()
        pos_fix[:, :, 1] = pos[:, :, 1] * (W / H)

        R = torch.cat((rcos, -rsin, rsin, rcos), 3).view(pos.shape[0], pos.shape[1], 2, 2)
        T = torch.cat((R*scale, pos_fix.unsqueeze(3)), 3)
        s = 64
        stacked = pyramid.stack(x,s,T,augment=train)

        self.stack_vis = stacked.detach()

        N = stacked.shape[0]

        batched = stacked.view(N*multi*self.levels,1,s,s)

        out = self.resnet(batched)

        out, self.heat_vis = mass2d(out)

        self.mass_vis = out.detach()

        out = torch.flatten(out,1)

        out = out.view(N*multi, -1)

        out = self.regressor(out)

        out = out.view(N,multi,1,2)

        out = torch.matmul(out,R.transpose(2,3)/scale)

        return out.view(N,multi,2)

def load_model(levels,name,load=False):
    model = PyramidAttention(levels)

    if load:
        model.load_state_dict(torch.load(f"Models/{name}.pt"))

    model.to('cuda')
    return model

def save_model(model, name):
    if not os.path.exists("Models"):
        os.mkdir("Models")
    torch.save(model.state_dict(), f"Models/{name}.pt")
