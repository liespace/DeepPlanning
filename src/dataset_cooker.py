import os
import shutil
import toolbox
import numpy as np

dir_parent = os.path.dirname(os.getcwd())
dir_food = dir_parent + '/food'
if not os.path.exists(dir_food):
    os.mkdir(dir_food)
    print('dir food is not exist, made it')

num_food = 0
states = []
paths = []
cdts = []
labels5 = []
labels4 = []
deltas5 = []
deltas4 = []
wastes = []
name_food_cld = '{}/food/cld.npz'.format(dir_parent)
name_food_spld = '{}/food/spld.npz'.format(dir_parent)

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

    label = np.asarray(toolbox.read_csv(name_label))
    cdt = np.asarray(toolbox.read_csv(name_cdt, delimiter=' '))
    delta = label - cdt

    cdts.append(cdt)

    states.append(cdt[0, :])
    paths.append(cdt[1:, :])

    labels5.append(label)
    labels4.append(label[1:, :])

    deltas5.append(delta)
    deltas4.append(delta[1:, :])

    name_gridmap2 = '{}/food/{}.png'.format(dir_parent, num_food)
    shutil.copy(name_gridmap, name_gridmap2)
    num_food += 1

np.savez(name_food_cld, cdts=cdts, labels=labels5, deltas=deltas5)
np.savez(name_food_spld, states=states, paths=paths, labels=labels4, deltas=deltas4)
