import os
import re
import numpy as np


def save_predictions(forecasts, cond='cond', root='dataset', folder='pred'):
    if not os.path.isdir(folder):
        os.makedirs(folder)
    with open(root + os.sep + cond + '.csv') as f:
        files = list(f)
    for i, forecast in enumerate(forecasts):
        # reshape prediction to 2D
        forecast = forecast.reshape((5, 4))
        # sigmoid the prediction
        forecast = 1 / (1 + np.exp(-forecast))
        # scale theta prediction back to original range
        forecast[:, -2] = forecast[:, -2] * 2 * np.pi - np.pi
        # scale x,y prediction back to original range
        forecast[:, 0:2] = forecast[:, 0:2] * 60 - 30
        # grip the no of cond
        no = re.sub('\D', '', files[i].strip().split(',')[0])
        # form the filename prediction
        name = no + '_pred.txt'
        np.savetxt(folder + os.sep + name, forecast, delimiter=',')
        # print ('saved ' + name)
    print ('saved ' + str(len(files)) + ' predictions ' +
           'to folder ' + os.getcwd() + os.sep + folder)
