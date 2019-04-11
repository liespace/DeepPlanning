import os
import toolbox
import numpy as np
from PIL import Image

dir_parent = os.path.dirname(os.getcwd())
dir_food = dir_parent + '/food'
if not os.path.exists(dir_food):
    os.mkdir(dir_food)
    print('dir food is not exist, made it')

gridmaps = []
states = []
paths = []
conditions = []
labels5 = []
labels4 = []
deltas5 = []
deltas4 = []
wastes = []
name_food_gcld = '{}/food/gcld.npz'.format(dir_parent)
name_food_gspld = '{}/food/gspld.npz'.format(dir_parent)

here = 0
end = 633
while here <= end:
    name_gridmap = '{}/dataset/{}gridmap.png'.format(dir_parent, here)
    name_cdt = '{}/dataset/{}condition.csv'.format(dir_parent, here)
    name_label = '{}/dataset/{}label.csv'.format(dir_parent, here)
    print(here)
    here += 1
    if not os.path.exists(name_label):
        wastes.append(name_label)
        print(name_label + ' is not exist')
        continue

    gridmap = np.array(Image.open(name_gridmap).convert(mode='RGB'), dtype=np.float32)
    label = np.asarray(toolbox.read_csv(name_label), dtype=np.float32)
    cdt = np.asarray(toolbox.read_csv(name_cdt, delimiter=' '), dtype=np.float32)

    if cdt.shape[0] > 5:
        print(name_cdt + 'is over size')
    if cdt.shape[0] < 5:
        print(name_cdt + 'is below size')
        while cdt.shape[0] < 5:
            cdt = np.append(cdt, [cdt[-1, :]], 0)
            label = np.append(label, [label[-1, :]], 0)
    if cdt.shape[0] < 5:
        print('is not enough')

    cdt, label = toolbox.gcs_to_lcs(cdt, label)

    delta = label - cdt

    gridmaps.append(gridmap)
    conditions.append(cdt)
    states.append(cdt[0, :])
    paths.append(cdt[1:, :])

    labels5.append(label)
    labels4.append(label[1:, :])

    deltas5.append(delta)
    deltas4.append(delta[1:, :])

np.savez(name_food_gcld, gridmaps=gridmaps, conditions=conditions, labels=labels5, deltas=deltas5)
np.savez(name_food_gspld, gridmaps=gridmaps, states=states, paths=paths, labels=labels4, deltas=deltas4)

# with np.load(name_food_gcld) as data:
#     feature = data['gridmaps']
#     print(feature.shape)
