import numpy as np
# from numpy.core.fromnumeric import transpose
import scipy.stats as stats
from scipy.io import loadmat

with open('Iris_TTT4275\class_1.txt', 'r') as f:
    x1all = np.array([[float(num) for num in line.split(',')] for line in f])
print("x1all: ", x1all)

with open('Iris_TTT4275\class_2.txt', 'r') as f:
    x2all = np.array([[float(num) for num in line.split(',')] for line in f])
print("x2all: ", x2all)

with open('Iris_TTT4275\class_3.txt', 'r') as f:
    x3all = np.array([[float(num) for num in line.split(',')] for line in f])
print("x3all: ", x3all)

# x2all = load('class_2', '-ascii')
# x3all = load('class_3', '-ascii')

# x1 = [x1all(:, 4) x1all(:, 1) x1all(:, 2)]
# x2 = [x2all(:, 4) x2all(:, 1) x2all(:, 2)]
# x3 = [x3all(:, 4) x3all(:, 1) x3all(:, 2)]

# x1 = [x1all(:, 3) x1all(:, 4)]
# x2 = [x2all(:, 3) x2all(:, 4)]
# x3 = [x3all(:, 3) x3all(:, 4)]
print("x1", x1all[:, 3])

x1 = np.array([x1all[:, 3]])
x2 = np.array([x2all[:, 3]])
x3 = np.array([x3all[:, 3]])
# x2 = np.empty(len(x2all))
# x3 = np.empty(len(x3all))

# for i in range(len(x1all)):
#     x1[i] = x1all[i][3]
#     x2[i] = x2all[i][3]

print(np.size(x1))
print("x1: ", x1)
# Ntot, dimx = np.size(x1)
Ntot = np.size(x1)

for k in range(1, 6):
    N1d = (k-1)*10 + 1
    print("N1d: ", N1d)
    N2d = np.remainder((k+2)*10-1, 50)+2
    print("N2d: ", N2d)
    N1t = np.remainder(N2d+1, 50)
    N2t = np.remainder(N2d+19, 50) + 1
    if(N2d < N1d):
        x1d = np.array([x1[N1d:Ntot, :], x1[0:N2d, :]])
        x2d = np.array([x2[N1d:Ntot, :], x2[0:N2d, :]])
        x3d = np.array([x3[N1d:Ntot, :], x3[0:N2d, :]])
    else:
        print(":(")
        # x1d = x1[:, N1d:N2d]
        x1d = x1[:, N1d:N2d]
        print("x1d: ", x1d)
        print("x1d length: ", len(x1d[0]))
        x2d = x2[:, N1d:N2d]
        x3d = x3[:, N1d:N2d]
    # end

    if(N2t < N1t):
        x1t = np.array([x1[N1t:Ntot, :], x1[0:N2t, :]])
        x2t = np.array([x2[N1t:Ntot, :], x2[0:N2t, :]])
        x3t = np.array([x3[N1t:Ntot, :], x3[0:N2t, :]])
    else:
        x1t = np.array(x1[N1t: N2t, :])
        x2t = np.array(x2[N1t: N2t, :])
        x3t = np.array(x3[N1t: N2t, :])
    # end

    Ndtot = 30
    Nttot = Ntot - Ndtot

    x1m = np.mean(x1d)
    print("x1m: ", x1m)
    x1s = np.std(x1d)
    print("x1s: ", x1s)
    x2m = np.mean(x2d)
    x2s = np.std(x2d)
    x3m = np.mean(x3d)
    x3s = np.std(x3d)

    indd = np.zeros((3, Ndtot))

    y1d = np.zeros((3, Ndtot))

    print("mvpdf", stats.multivariate_normal.pdf(
        x1d[0], [x1m], [x1s], allow_singular=True))
    y1d[0] = stats.multivariate_normal.pdf(x1d[0], x1m, x1s)
    y1d[1] = stats.multivariate_normal.pdf(x1d[0], x2m, x2s)
    y1d[2] = stats.multivariate_normal.pdf(x1d[0], x3m, x3s)
    val1d, indd[:, 1-1] = np.maximum(np.transpose(y1d))

    y2d = np.zeros((Ndtot, 3))
    y2d[:, 0] = stats.multivariate_normal.pdf(x2d[0], x1m, x1s)
    y2d[:, 1] = stats.multivariate_normal.pdf(x2d[0], x2m, x2s)
    y2d[:, 2] = stats.multivariate_normal.pdf(x2d[0], x3m, x3s)
    val2d, indd[:, 2-1] = np.maximum(np.transpose(y2d))

    y3d = np.zeros((Ndtot, 3))
    y3d[:, 0] = stats.multivariate_normal.pdf(x3d[0], x1m, x1s)
    y3d[:, 1] = stats.multivariate_normal.pdf(x3d[0], x2m, x2s)
    y3d[:, 2] = stats.multivariate_normal.pdf(x3d[0], x3m, x3s)
    val3d, indd[:, 3-1] = np.maximum(np.transpose(y3d))

    confd = np.zeros((3, 3))

    for i in range(1-1, 4-1):  # correct class
        for j in range(1-1, 4-1):  # chosen class
            confd[i, j] = len(np.find(indd[:, i] == j))
    # end
    # end

    indt = np.zeros((Nttot, 3))

    y1t = np.zeros((Nttot, 3))
    y1t[:, 1-1] = stats.multivariate_normal.pdf(x1t, x1m, x1s)
    y1t[:, 2-1] = stats.multivariate_normal.pdf(x1t, x2m, x2s)
    y1t[:, 3-1] = stats.multivariate_normal.pdf(x1t, x3m, x3s)
    val1t, indt[:, 1-1] = np.maximum(np.transpose(y1t))

    y2t = np.zeros((Nttot, 3))
    y2t[:, 1-1] = stats.multivariate_normal.pdf(x2t, x1m, x1s)
    y2t[:, 2-1] = stats.multivariate_normal.pdf(x2t, x2m, x2s)
    y2t[:, 3-1] = stats.multivariate_normal.pdf(x2t, x3m, x3s)
    val2t, indt[:, 2-1] = np.maximum(np.transpose(y2t))

    y3t = np.zeros((Nttot, 3))
    y3t[:, 1-1] = stats.multivariate_normal.pdf(x3t, x1m, x1s)
    y3t[:, 2-1] = stats.multivariate_normal.pdf(x3t, x2m, x2s)
    y3t[:, 3-1] = stats.multivariate_normal.pdf(x3t, x3m, x3s)
    val3t, indt[:, 3-1] = np.maximum(np.transpose(y3t))

    conft = np.zeros((3, 3))

    for i in range(1-1, 4-1):  # correct class
        for j in range(1-1, 4-1):  # chosen class
            conft[i, j] = len(np.where(indt[:, i] == j))
            # end
    # end
    print([confd, conft])
    # end
