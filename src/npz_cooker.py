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
gridmaps_vd = []
paths = []
paths_vd = []
conditions = []
conditions_vd = []
labels5 = []
labels5_vd =[]
labels4 = []
labels4_vd = []
deltas5 = []
deltas5_vd = []
deltas4 = []
deltas4_vd = []
wastes = []
name_food_gcld = '{}/food/gcld_train.npz'.format(dir_parent)
name_food_gpld = '{}/food/gpld_train.npz'.format(dir_parent)
name_food_gcld_vd = '{}/food/gcld_validation.npz'.format(dir_parent)
name_food_gpld_vd = '{}/food/gpld_validation.npz'.format(dir_parent)

here = 0
end = 633
num = 0
step = 10
td_num = 0
vd_num = 0
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

    # if num == 392:
    #     print("it is you: {}".format(here))
    num += 1

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

    if num % step == 0:
        gridmaps_vd.append(gridmap)

        conditions_vd.append(cdt)
        paths_vd.append(cdt[1:, :])

        labels5_vd.append(label)
        labels4_vd.append(label[1:, :])

        deltas5_vd.append(delta)
        deltas4_vd.append(delta[1:, :])

        vd_num += 1
        continue

    gridmaps.append(gridmap)

    conditions.append(cdt)
    paths.append(cdt[1:, :])

    labels5.append(label)
    labels4.append(label[1:, :])

    deltas5.append(delta)
    deltas4.append(delta[1:, :])

    td_num += 1

print('cooking...')
np.savez(name_food_gcld, gridmap=gridmaps, condition=conditions, label=labels5, delta=deltas5)
np.savez(name_food_gpld, gridmap=gridmaps, condition=paths, label=labels4, delta=deltas4)
np.savez(name_food_gcld_vd, gridmap=gridmaps_vd, condition=conditions_vd, label=labels5_vd, delta=deltas5_vd)
np.savez(name_food_gpld_vd, gridmap=gridmaps_vd, condition=paths_vd, label=labels4_vd, delta=deltas4_vd)
print('cook {} examples'.format(num))
print('cook {} train examples'.format(td_num))
print('cook {} validation examples'.format(vd_num))

# with np.load(name_food_gcld) as data:
#     feature = data['gridmaps']
#     print(feature.shape)
