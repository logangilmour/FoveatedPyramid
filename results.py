import numpy as np
from math import sqrt
import sys
ver = int(sys.argv[1])
if ver==1:
    with open('results_lil_1.npz', 'rb') as f:
        res = np.load(f)['arr_0']

    with open('doctors_lil.npz', 'rb') as f:
        doc = np.load(f)['arr_0']
elif ver==0:
    with open('results_lil_fixed_0.npz', 'rb') as f:
        res = np.load(f)['arr_0']
    with open('doctors_lil2.npz', 'rb') as f:
        doc = np.load(f)['arr_0']

elif ver==2:
    errors = []
    run = 0
    for i in range(19):
        with open(f'Results/lil_random_{i}_{run}.npz', 'rb') as f:
            err = np.load(f)['arr_0']
            errors.append(err)
    res = np.stack(errors).reshape(1,19,150,1)

    with open('doctors_lil.npz', 'rb') as f:
        doc = np.load(f)['arr_0']

elif ver==3:
    all_errors = []
    for run in range(4):
        errors = []
        for i in range(19):
            with open(f'Results/big_hybrid_{i}_{run}.npz', 'rb') as f:
                err = np.load(f)['arr_0']
                errors.append(err)
        all_errors.append(np.stack(errors))
    res = np.stack(all_errors)

    with open('doctors_big.npz', 'rb') as f:
        doc = np.load(f)['arr_0']

elif ver==4:
    errors = []
    run = 0
    for i in range(19):
        with open(f'Results/lil_hybrid_{i}_{run}.npz', 'rb') as f:
            err = np.load(f)['arr_0']
            errors.append(err)
    res = np.stack(errors).reshape(1,19,150,1)

    with open('doctors_lil.npz', 'rb') as f:
        doc = np.load(f)['arr_0']

elif ver==5:
    errors = []
    with open('results_lil_hybrid_test2_0.npz', 'rb') as f:
        res = np.load(f)['arr_0']

    with open('doctors_lil2.npz', 'rb') as f:
        doc = np.load(f)['arr_0']


print(res.shape)

names = [
"Sella (L1)",
"Nasion (L2)",
"Orbitale (L3)",
"Porion (L4)",
"Subspinale (L5)",
"Supramentale (L6)",
"Pogonion (L7)",
"Menton (L8)",
"Gnathion (L9)",
"Gonion (L10)",
"Incision inferius (L11)",
"Incision superius (L12)",
"Upper lip (L13)",
"Lower lip (L14)",
"Subnasale (L15)",
"Soft tissue pogonion (L16)",
"Posterior nasal spine (L17)",
"Anterior nasal spine (L18)",
"Articulare (L19)",]

str = "Landmark & PEL (mm) & IOV (mm) & SDR 2.0mm & SDR 2.5mm & SDR 3.0mm & SDR 4.0mm\\\\\n"

str+="\\hline\n"


res = res.transpose(1,0,2,3)
doc = doc.transpose(1,0,2,3)
numel = res.shape[2]
for i, r in enumerate(res):
    d = doc[i]
    rm = f"{r.mean():2.2f}"
    dm = f"{d.mean():2.2f}"
    if r.mean()<d.mean():
        rm = "\\textbf{"+rm+"}"
    else:
        dm = "\\textbf{"+dm+"}"

    str += f"{names[i]} & {rm} $\\pm$ {(r.std(1)).mean():2.2f} & {dm} $\\pm$ {(d.std(1)).mean():2.2f} & {((r < 2).sum(1) / numel * 100).mean():3.2f} & {((r < 2.5).sum(1) / numel * 100).mean():3.2f} & {((r < 3).sum(1) / numel * 100).mean():3.2f} & {((r < 4).sum(1) / numel * 100).mean():3.2f}\\\\\n"
str+="\hline\n"
str+=f"Average & \\textbf{{{res.mean():2.2f}}} $\\pm$ {(res.std(2)).mean():2.2f} & {doc.mean():2.2f} $\\pm$ {(doc.std(2)).mean():2.2f} & {((res<2).sum(2)/numel*100).mean():3.2f} & {((res<2.5).sum(2)/numel*100).mean():3.2f} & {((res<3).sum(2)/numel*100).mean():3.2f} & {((res<4).sum(2)/numel*100).mean():3.2f}\\\\\n"

print(str)



