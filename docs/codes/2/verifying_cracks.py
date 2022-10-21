# Implementation to detect cracks through image processing

import numpy as np

val = np.array([3,7,4,5,6,2,8,9,10])
print(val.mean())

diff = np.array([abs(i-val.mean()) for i in val])
print(diff)

if max(diff) > val.mean():
    print('Existe rachadura')
else:
    print('Não existe rachadura')
