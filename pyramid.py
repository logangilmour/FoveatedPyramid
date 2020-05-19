import math
import torch
import torch.nn.functional as F
import numpy as np

def pyramid(I, levels):
    """

    :rtype: object
    """
    pym = []
    for i in range(levels-1):
        pym.append(I)
        I=gaussian_reduce(I,sigma=2.0)

    pym.append(I)

    return pym


def pyramid_transform(T,H,W, size, level):
    i = level
    scale = torch.tensor([
        size / W, size / W, 1, size / H, size / H, 1], dtype=torch.float32,
        device=T.device).reshape(2, 3)*torch.tensor([2 ** i, 2 ** i, 1], device=T.device)
    T = T*scale
    M = T[:,:2]
    Minv = M.inverse()
    t = T[:,[2]]
    T = torch.cat(
        (Minv,torch.mm(-Minv,t)),1
    )
    return T

def stack(pym, size, T, augment=False):
    N = pym[0].shape[0]
    C = T.shape[1]
    H = pym[0].shape[2]
    W = pym[0].shape[3]
    stacked = torch.zeros(N,C,len(pym),size,size,device=pym[0].device)

    sample = torch.arange(-(size-1)/size,1,2/size,device=pym[0].device)
    gy,gx = torch.meshgrid(sample,sample)
    grid = torch.stack((gx,gy,torch.ones((size,size),device=pym[0].device)),2).expand(N,C,size,size,3)

    scale = torch.tensor([size / W, size / W, 1, size / H, size / H, 1], dtype=torch.float32,
                         device=pym[0].device).reshape(1, 1, 2, 3)
    for i in range(len(pym)):
        Tl = T * torch.tensor([2 ** i, 2 ** i, 1], device=pym[0].device) * scale

        g = torch.matmul(grid.view(N,C,size*size,3), Tl.transpose(2,3))\
            .view(N,size*C,size,2)
        stacked[:, :, i, :, :] = F.grid_sample(pym[i],g,align_corners=False).view(N,C,size,size)

    if augment:
        stacked+=torch.randn(stacked.shape,device=pym[0].device)*0.01
    return stacked


def box_reduce(I):
    return (I[:,:, ::2, ::2] + I[:,:, 1::2, ::2] + I[:,:, ::2, 1::2] + I[:,:, 1::2, 1::2]) * 0.25

def gaussian_reduce(input, sigma=2.0/6.0, truncate=3.0):
    reduction = 0.5
    sd = math.sqrt(2 * math.log(1 / 0.5)) / (math.pi * reduction)
    size = input.shape
    #input = torch.reshape(input,(-1,1,size[2],size[3]))
    # make the radius of the filter equal to truncate standard deviations
    lw = int(truncate * sd + 0.5)
    weights = [0.0] * (2 * lw + 1)
    weights[lw] = 1.0
    sum = 1.0
    sd = sd * sd
    # calculate the kernel:
    for ii in range(1, lw + 1):
        tmp = math.exp(-0.5 * float(ii * ii) / sd)
        weights[lw + ii] = tmp
        weights[lw - ii] = tmp
        sum += 2.0 * tmp
    for ii in range(2 * lw + 1):
        weights[ii] /= sum
    wH = torch.tensor(weights).reshape(1,1,1,2*lw+1).to(input.device)
    wV = torch.tensor(weights).reshape(1,1,2*lw+1,1).to(input.device)

    with torch.set_grad_enabled(False):
        out = F.conv2d(input,wH,padding=(0,lw))
        out = F.conv2d(out,wV,padding=(lw,0),stride=(2,2))

        #out = torch.reshape(out,(-1,3,out.shape[2],out.shape[3]))
        return out

def test():
    from time import time
    from XrayData import HeadXrays, Transform
    from torchvision import transforms
    import matplotlib.pyplot as plt
    from math import sin, cos, pi

    trans = transforms.Compose([transforms.ToTensor(), lambda x: x[:, :, :1920].sum(dim=0, keepdim=True),
                                transforms.Normalize([1.4255656], [0.8835338])])

    data = Transform(HeadXrays("images/RawImage"), tx=trans)

    dataloader = torch.utils.data.DataLoader(data,batch_size=2, shuffle=False, num_workers=4,
                                                  pin_memory=True)

    test = next(iter(dataloader))[0].cuda()
    batch=2
    heads = 19
    sz = 128
    reps = 3
    levels = 6
    #test = torch.zeros((batch, 1, 1920, 2400)).cuda()
    start = time()
    for i in range(1):#range(324 // batch):
        start_level = 1
        level = 3
        pym = pyramid(test, start_level,levels)

        for j in range(1):#reps*19):
            #storage = torch.zeros(batch, heads, levels, sz, sz)
            #for i in range(heads):
            theta = pi/4
            T = torch.tensor([
                [[cos(theta), -sin(theta), 1],
                 [sin(theta), cos(theta), 0]],
                [[1, 0, -1],
                 [0, 1, -1920/2432.]]
                ],device='cuda').expand(batch,-1,-1,-1)
            s = stack(pym, sz, T)
                #storage[:, i,:,:,:] = s


            #offset = torch.tensor([0,0,0,0,0,0],device='cuda').reshape(2,3)

            TR = T[0,0]
            t = pyramid_transform(TR,2432,1920,sz,start_level,level)
            #print(s.shape)
            print(time() - start)
            print(s.shape)
            plt.imshow(s[0, 0, level, :, :].squeeze().cpu().numpy())
            pnt = torch.tensor([1,-1,1.],device='cuda').reshape(1,3)
            #up = torch.matmul(pnt,ts[0,0,level-1,:,:].t())
            #print(up.shape,T[:,:,:,[2].shape)
            #T[:, :, :, [2]] = up.reshape(1,1,2,1)
            #print(T.shape)
            #s, ts = stack(pym,sz,T)
            #plt.imshow(s[0, 0, level, :, :].squeeze().cpu().numpy())
            H=2432.
            W=1920.
            pnt = torch.matmul(pnt,t.t())
            #pnt[:,1] = pnt[:,1]*(H/W)
            screen = (pnt.cpu().numpy()+1)*0.5*sz
            #print(screen)
            plt.plot(screen[:,0],screen[:,1],"r.")
            plt.show()

if __name__=='__main__':
    test()


