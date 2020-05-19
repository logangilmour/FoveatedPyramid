import XrayData
import torchvision.transforms as transforms
import numpy as np
from random import randint
import torch
import model as m
import torch.optim as optim
import torch.nn as nn
from time import time
from math import pi
import matplotlib
import copy
import torch.nn.functional as F
from pyramid import pyramid, stack, pyramid_transform
import sys

def all():
    folds_errors = []
    for fold in range(4):
        errors = []
        for i in range(19):
            path = f"Models/big_{i}_{fold}.pt"
            errors.append(test([{'loadpath':path}],[i],fold=fold))
        all_errors = np.stack(errors)
        folds_errors.append(all_errors)
    all_folds_errors = np.stack(folds_errors)
    print(all_errors.mean())
    with open(f'results_big.npz', 'wb') as f:
        np.savez(f, all_folds_errors)


def test(settings, landmarks,fold=3, num_folds =4, fold_size=100, avg_labels=True):
    print("TEST")


    batchsize=2
    device = 'cuda'

    splits, datasets, dataloaders, _ = XrayData.get_folded(landmarks,batchsize=batchsize, fold=fold, num_folds=num_folds, fold_size=fold_size)

    annos = XrayData.TransformedHeadXrayAnnos(indices=list(range(150)), landmarks=landmarks)

    if avg_labels:
        pnts = np.stack(list(map(lambda x: (x[1] + x[2]) / 2, annos)))
    else:
        pnts = np.stack(list(map(lambda x: x[1], annos)))

    means = torch.tensor(pnts.mean(0, keepdims=True), device=device, dtype=torch.float32)

    levels = 6

    output_count=len(landmarks)



    models = []
    for setting in settings:
        model = m.PyramidAttention(levels)
        model.load_state_dict(torch.load(setting['loadpath']))
        models.append(model)
        model.to(device)
        model.eval()

    criterion = nn.MSELoss(reduction='none')
    # Iterate over data.

    phase='val'
    data_iter = iter(dataloaders[phase])
    next_batch = data_iter.next()  # start loading the first batch

    # with pin_memory=True and async=True, this will copy data to GPU non blockingly
    next_batch = [t.cuda(non_blocking=True) for t in next_batch]

    start = time()
    errors = []
    doc_errors = []
    print("GOT HERE")
    for i in range(len(dataloaders[phase])):
        batch = next_batch
        inputs, junior_labels, senior_labels = batch


        if i + 2 != len(dataloaders[phase]):
            # start copying data of next batch
            next_batch = data_iter.next()
            next_batch = [t.cuda(non_blocking=True) for t in next_batch]


        inputs_tensor = inputs.to(device)

        if avg_labels:
            labels_tensor = torch.stack((junior_labels, senior_labels), dim=0).mean(0).to(device).to(torch.float32)
        else:
            labels_tensor = junior_labels.to(device).to(torch.float32)

        # zero the parameter gradients

        pym = pyramid(inputs_tensor, levels)

        # forward
        # track history if only in train
        with torch.set_grad_enabled(False):
            # Get model outputs and calculate loss
            all_outputs = []
            for model in models:
                guess = means


                for j in range(10):

                    outputs = guess + model(pym, guess,
                                            phase == 'train')  # ,j==2 and i==0 and phase=='val' and False,rando)




                    guess = outputs.detach()

                all_outputs.append(guess)

            avg = torch.stack(all_outputs,0).mean(0)

            loss = criterion(avg, labels_tensor)

            error = loss.detach().sum(dim=2).sqrt()
            errors.append(error)
            doc_errors.append(F.mse_loss(junior_labels, senior_labels, reduction='none').sum(dim=2).sqrt())

    errors = torch.cat(errors,0).detach().cpu().numpy()/2*192
    doc_errors = torch.cat(doc_errors,0).detach().cpu().numpy()/2*192

    doc_error = doc_errors.mean(0)
    all_error = errors.mean(0)
    error = errors.mean()
    for i in range(output_count):

        print(f"Error {i}: {all_error[i]} (doctor: {doc_error[i]}")

    print(f"{phase} loss: {error} (doctors: {doc_errors.mean()} in: {time() - start}")
    return errors



if __name__=='__main__':
    if len(sys.argv)>1:


        test_num = int(sys.argv[1])
        if test_num==1:
            folds_errors = []
            fold = 1

            errors = []
            run = 0
            from time import time
            rt = time()
            for i in range(19):
                settings = []
                #for run in range(1,2):
                path = f"Models/lil_hybrid_{i}_{run}.pt"
                settings.append({'loadpath': path})
                errors.append(test(settings, [i], fold=3,num_folds=4,fold_size=100))
            all_errors = np.stack(errors)
            folds_errors.append(all_errors)

            all_folds_errors = np.stack(folds_errors)
            print(all_errors.mean())
            with open(f'results_lil_hybrid_test2_{run}.npz', 'wb') as f:
                np.savez(f, all_folds_errors)
            print(time()-rt)
        else:
            all()

    else:
        print(test([{'loadpath':"Models/test.pt"}],[11]).mean())

'''
if __name__ == '__main__':
    errors = []

    #for i in range(19):
    #    errors.append(test([], [i]))

    for i in range(17):
        errors.append(test([{'loadpath':f"Models_seed_10/single_{i}.pt",'fuckup_override':1},
                            {'loadpath': f"Models_seed_20/single_{i}.pt", 'fuckup_override': 1},
                            {'loadpath': f"Models_seed_100/single_{i}.pt", 'fuckup_override': 2},
                            {'loadpath': f"Models_seed_30/single_{i}.pt", 'fuckup_override': 0}
                            ], [i]))
    for i in range(17,19):
        errors.append(test([
            {'loadpath':f"Models_seed_10/single_{i}.pt",'fuckup_override':1},
            {'loadpath': f"Models_seed_20/single_{i}.pt", 'fuckup_override': 1},
             {'loadpath': f"Models_seed_100/single_{i}.pt", 'fuckup_override': 0},

            {'loadpath': f"Models_seed_30/single_{i}.pt", 'fuckup_override': 0},
                            ], [i]))

    all_errors = np.stack(errors)
    print(all_errors.mean())
    with open('results_ensemble.npz', 'wb') as f:
        np.savez(f, all_errors)
'''

