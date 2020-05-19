import XrayData
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import numpy as np
from random import randint
import torch
import model as m
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
from time import time
from math import pi
import matplotlib
import os
import sys
import copy
import torch.nn.functional as F
from pyramid import pyramid, stack, pyramid_transform

def all():
    errors=[]
    for i in range(19):
        errors.append(train(f"single_{i}",[i]))
    print(errors)


def train(name, landmarks, load=False, startEpoch=0, batched=False, fold=3, num_folds=4, fold_size=100, graphical=False,iterations=10, avg_labels=False,rms=False):
    print(f"AVG? {avg_labels}, RMS? {rms}")
    print(f"BEGIN {name} {landmarks}")
    batchsize=2
    num_epochs=40
    device = 'cuda'


    splits, datasets, dataloaders, annos = XrayData.get_folded(landmarks,batchsize=batchsize,fold=fold,num_folds=num_folds,fold_size=fold_size)



    if avg_labels:
        pnts = np.stack(list(map(lambda x: (x[1]+x[2])/2, annos)))
    else:
        pnts = np.stack(list(map(lambda x: x[1], annos)))

    means = torch.tensor(pnts.mean(0,keepdims=True),device=device,dtype=torch.float32)
    stddevs = torch.tensor(pnts.std(0,keepdims=True),device=device,dtype=torch.float32)
    levels = 6


    # TODO goes with plotting
    if graphical:
        matplotlib.rcParams['figure.figsize'] = [18, 48/19]


    model = m.load_model(levels, name, load)



    best_error = 1000
    last_error = 1000

    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    for ep in range(num_epochs):
        epoch = startEpoch+ep

        if epoch>=20:
            for g in optimizer.param_groups:
                g['lr']= 0.00001

        for i, g in enumerate(optimizer.param_groups):
            print(f"LR {i}: {g['lr']}")

        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)


        if graphical:
            fix, ax = plt.subplots(1, 3)
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                if batched and epoch < num_epochs-1:
                    continue
                model.eval()   # Set model to evaluate mode


            rando = randint(0,len(dataloaders[phase]))

            # TODO could this be wrapped up somehow?
            data_iter = iter(dataloaders[phase])
            next_batch = data_iter.next()  # start loading the first batch

            # with pin_memory=True and async=True, this will copy data to GPU non blockingly
            next_batch = [t.cuda(non_blocking=True) for t in next_batch]

            start = time()

            # TODO error tracking should be its own thing maybe
            errors = []
            doc_errors = []
            for i in range(len(dataloaders[phase])):
                batch = next_batch
                inputs,junior_labels, senior_labels = batch

                if i + 2 != len(dataloaders[phase]):
                    # start copying data of next batch
                    next_batch = data_iter.next()
                    next_batch = [t.cuda(non_blocking=True) for t in next_batch]


                inputs_tensor = inputs.to(device)

                if avg_labels:
                    labels_tensor = torch.stack((junior_labels,senior_labels),dim=0).mean(0).to(device).to(torch.float32)
                else:
                    labels_tensor = junior_labels.to(device).to(torch.float32)
                # zero the parameter gradients

                pym = pyramid(inputs_tensor, levels)

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    if phase == 'train':
                        guess = torch.normal(means.expand(2,2,2), stddevs.expand(2,2,2)).to(device)
                    else:
                        guess = means

                    for j in range(iterations):
                        #if phase == 'train':
                        #    guess = torch.normal(means, stddevs).to(device).clamp(-1, 1)
                        optimizer.zero_grad()
                        outputs = guess + model(pym, guess, phase=='train')
                        loss = F.mse_loss(outputs, labels_tensor,reduction='none')

                        if phase == 'train':
                            if rms:
                                F.mse_loss(outputs, labels_tensor, reduction='none').sum(dim=2).sqrt().mean().backward()
                            else:
                                F.l1_loss(outputs, labels_tensor, reduction='mean').backward()

                            optimizer.step()

                        guess = outputs.detach()
                        #if phase=='train':
                            #guess+=torch.randn(guess.shape,device=device)*(max(8-j,0))/100

                    error = loss.detach().sum(dim=2).sqrt()
                    errors.append(error)
                    doc_errors.append(F.mse_loss(junior_labels,senior_labels,reduction='none').sum(dim=2).sqrt())

                # TODO this has got to move somewhere else
                if i==rando//batchsize and phase=='val' and graphical:
                    res = 32
                    pos = labels_tensor.detach().clone()

                    theta = (torch.rand((pos.shape[0], pos.shape[1], 1, 1), device=pym[0].device)*2-1)*pi/6
                    #theta = theta*0
                    rsin = theta.sin()
                    rcos = theta.cos()
                    R = torch.cat((rcos, -rsin, rsin, rcos), 3).view(pos.shape[0], pos.shape[1], 2, 2)

                    H = 2432
                    W = 1920

                    pos[:, :, 1] = pos[:, :, 1] * (W / H)



                    T = torch.cat((
                        #torch.eye(2, device=pym[0].device).expand(pos.shape[0], pos.shape[1], 2, 2),
                        R,
                        pos.unsqueeze(3)), 3)



                    stacked = stack(pym,res,T)

                    index = rando%batchsize


                    plots = [labels_tensor[index], junior_labels[index], senior_labels[index],outputs[index]]

                    level = 2
                    pid =0


                    ax[0].imshow(stacked[index, pid, level, :, :].cpu().numpy())
                    for c, p in enumerate(plots):
                        p = p.to(torch.float32).clone()
                        p[:, 1] = p[:, 1] * (W / H)
                        #



                        TR = pyramid_transform(T[index,pid],H,W,res,level)

                        pnt = torch.cat((p,torch.ones((p.shape[0],1),device=device)),1)
                        pnt = torch.matmul(pnt, TR.t())
                        screen = (pnt.cpu().numpy()+1)*0.5*res

                        styles = ["r","m","b","w"]

                        ax[0].plot(screen[pid, 0], screen[pid, 1], "." + styles[c])



            # TODO more error code separation
            errors = torch.cat(errors,0).detach().cpu().numpy()/2*192
            doc_errors = torch.cat(doc_errors,0).detach().cpu().numpy()/2*192


            error = errors.mean()

            # TODO Draw code should be separate
            if graphical:
                ax[1 if phase == 'train' else 2].hist(errors[:, 0], bins=30)


            if phase == 'train':
                if not batched or epoch == num_epochs-1:
                    m.save_model(model, name)
                last_error = error

            if phase == 'val' and error < best_error:
                best_error = error
                print(f"New best {error}")

            if phase == 'val' and batched:
                if not os.path.exists("Results"):
                    os.mkdir("Results")
                with open(f'Results/{name}.npz', 'wb') as f:
                    np.savez(f, errors)

            print(f"{phase} loss: {error} (doctors: {doc_errors.mean()} in: {time() - start}")
        if graphical:
            plt.show(block=False)

    return last_error,best_error


if __name__ == '__main__':

    if len(sys.argv)>1:
        test = int(sys.argv[1])

        id = int(sys.argv[2])
        if test==0:
            print("RUNNING LARGE TEST")
            fold = id//10
            pnt = id % 10 * 2

            for i in range(pnt,min(pnt+2,19)):
                print(f"Running fold {fold}, point {i}")
                train(f"big_hybrid_{i}_{fold}", [i],batched=True,fold=fold,num_folds=4,fold_size=100,iterations=10,avg_labels=False)
        elif test==1:
            print("RUNNING SMALL TEST")
            fold = 1
            pnt = id % 5 * 4
            run = id//5

            for i in range(pnt, min(pnt + 4, 19)):
                print(f"Running fold {fold}, point {i}")
                train(f"lil_hybrid_{i}_{run}", [i], batched=True, fold=fold, num_folds=2, fold_size=150,iterations=10,avg_labels=True)
    else:
        train("test_avg_up",[0],num_folds=2,fold_size=150,fold=1,graphical=False,avg_labels=False,iterations=10,batched=True)#,load=True,startEpoch=20)
