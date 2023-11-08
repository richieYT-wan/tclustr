import sys
from torch import cuda

if cuda.is_available():
    print('CUDA HERE')
    sys.exit(0)
else:
    print('NO CUDA HERE!!')
    sys.exit(1)
