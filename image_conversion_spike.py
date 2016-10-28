from PIL import Image

import scipy.io as sio
from numpy.testing import rand


def get():
    datadict = sio.loadmat('/Users/athakur/Downloads/test_32x32.mat')
    y = datadict['y'].reshape(datadict['y'].shape[0], )
    return datadict['X'].transpose((3, 0, 1, 2)), y

if __name__=="__main__":
    X,Y = get()
    for data in X[0:2]:
        img = Image.fromarray(data, 'RGB')
        img.save('my'+str(rand(0))+'.png')
        print(Y[0])
        img.show()