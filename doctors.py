from XrayData import HeadXrayAnnos
import numpy as np


annos = HeadXrayAnnos("images/RawImage")

errors=[]

for path, junior, senior in annos[:400]:
    errors.append(np.sqrt(((junior/10-senior/10)**2).sum(1))/2)


all_errors = np.stack(errors).transpose(1,0).reshape(1,19,400,1)

with open(f'doctors_big.npz', 'wb') as f:
    np.savez(f, all_errors)

errors=[]

for path, junior, senior in annos[150:300]:
    errors.append(np.sqrt(((junior/10-senior/10)**2).sum(1))/2)


all_errors = np.stack(errors).transpose(1,0).reshape(1,19,150,1)

with open(f'doctors_lil.npz', 'wb') as f:
    np.savez(f, all_errors)

errors=[]

for path, junior, senior in annos[300:400]:
    errors.append(np.sqrt(((junior/10-senior/10)**2).sum(1))/2)


all_errors = np.stack(errors).transpose(1,0).reshape(1,19,100,1)

with open(f'doctors_lil2.npz', 'wb') as f:
    np.savez(f, all_errors)
