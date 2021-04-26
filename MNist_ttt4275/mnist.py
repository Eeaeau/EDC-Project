from os.path import dirname, join as pjoin
import scipy.io as sio


#data_dir = pjoin(dirname(sio.__file__), 'Desktop', 'estimering', 'Mnist')
#mat_fname = pjoin(data_dir, 'data_all.mat')
mat_contents = sio.loadmat("Mnist_ttt4275/data_all.mat")
print(mat_contents.keys())
#print(mat_contents['col_size'])
print(mat_contents['testv'])