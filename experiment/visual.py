from __future__ import print_function
import os
import numpy as np
import glob
import matplotlib.pyplot as plt

def plot_pdf_cdf(x):
    # mean = 100
    # sigma = 1
    # x = mean + sigma * np.random.randn(10000)
    fig, (ax0, ax1) = plt.subplots(nrows=2, figsize=(9, 6))
    ax0.hist(x, 100, weights=np.ones_like(x) / float(len(x)),
             histtype='step', facecolor='yellowgreen', alpha=0.75)
    ax0.set_title('pdf')
    ax1.hist(x, 100, normed=1, histtype='step', facecolor='pink', alpha=0.75,
             cumulative=True, rwidth=0.8)
    ax1.set_title("cdf")
    fig.subplots_adjust(hspace=0.4)
    plt.show()

threshold = 0.6
coor_mask = np.array([0.8, 0.2])
root_pred = 'pred/valid'
root_true = 'dataset' + os.sep + 'well'
folders = os.listdir(root_pred)
for folder in folders[0:1]:
    path_pred = root_pred + os.sep + folder
    print("Processing " + path_pred)
    files = glob.glob(path_pred + os.sep + '*.txt')
    rights = 0.
    op_bias = 0.
    om_bias = 0.
    x_error = []
    y_error = []
    t_error = []
    for f in files:
        # number of the example
        no = f.split('/')[-1].split('_')[0]
        # ground true
        file_true = root_true + os.sep + str(no) + '_way.txt'
        true = np.loadtxt(file_true, delimiter=',')
        # grip the number of points
        n_prt = true.shape[0] - 2
        # grip the points
        prt = true[1:-1, :-1]
        # prediction
        pred = np.loadtxt(f, delimiter=',')
        p_obj = pred[:, -1]
        # grip the number of predicted points
        n_obj = sum(p_obj > threshold)
        # grip the predicted points
        p_prt = []
        for row in pred:
            if row[-1] > threshold:
                p_prt.append(row[:-1])
        # check if the numbers of points are equal
        rights += n_obj == n_prt
        op_bias += (n_obj - 1) == n_prt
        om_bias += (n_obj + 1) == n_prt
        # calculate the coordinate error of points
        if n_obj == n_prt and prt.shape[0] > 0:
            q = prt - np.array(p_prt)
            for i,r in enumerate(q):
                if np.abs(r[0] > 9):
                    print(prt, np.array(p_prt), f)
            x_error.extend(q[:, 0])
            y_error.extend(q[:, 1])
            t_error.extend(q[:, 2])
        # if prt.shape[0] == 0:
        #     continue
        # for p in p_prt:
        #     q = prt - p
        #     dxy = np.sqrt(q[:, 0]**2 + q[:, 1]**2)
        #     dt = np.abs(q[:, -1])
        #     ds = dxy * coor_mask[0] + dt * coor_mask[-1]
        #     index = np.argmin(ds)
        #     x_error.append(q[index][0])
        #     y_error.append(q[index][1])
        #     t_error.append(q[index][2])
    # print(x_error, y_error, t_error)
    print(np.mean(x_error), np.var(x_error), np.max(x_error), np.min(x_error))
    print(np.mean(y_error), np.var(y_error), np.max(y_error), np.min(y_error))
    print(np.mean(t_error), np.var(t_error), np.max(t_error), np.min(t_error))
    plot_pdf_cdf(x_error)
    break
    # stochastic performance
    precision = rights / len(files) * 100
    ob_precision = (rights + op_bias + om_bias) / len(files) * 100
    print(precision, ob_precision, rights, op_bias, om_bias)


