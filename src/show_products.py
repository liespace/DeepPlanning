import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from numpy.matlib import repmat
import numpy as np
import csv
import os


def read_csv(file_name, delimiter=','):
    products = csv.reader(open(file_name, newline=''), delimiter=delimiter, quotechar='|')
    my_list = []
    for row in products:
        dozen = []
        for item in row:
            if item is not '':
                dozen.append(float(item))
        my_list.append(dozen)
    return my_list


def where(benchmark, your_list):
    my_list = []
    for index, item in enumerate(your_list):
        if item > benchmark:
            my_list.append(index)
    return my_list


def main():
    fig = plt.gcf()
    ax = fig.add_subplot(1, 1, 1)
    plt.axis("equal")
    fig.show()
    fig.canvas.draw()

    no = 530
    parent_folder = os.path.dirname(os.getcwd())
    sproducts_file_name = '{}/{}sproducts.csv'.format(parent_folder, no)
    image_file_name = '{}/{}gridmap.png'.format(parent_folder, no)
    condition_file_name = '{}/{}condition.csv'.format(parent_folder, no)

    # Show Conditions ##########################################################

    # Read conditions
    conditions = np.asarray(read_csv(condition_file_name, ' '))
    origin = conditions[0, 0:2]
    theta = -conditions[0][2]
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    rect = plt.Rectangle((-5 / 2, -3 / 2), 5, 3, fill=True, edgecolor='red', linewidth=1)
    ax.add_patch(rect)

    path = conditions[1:, 0:2]
    headings = conditions[1:, 2]
    print(headings.shape)
    origins = repmat(np.r_[origin], path.shape[0], 1)
    path -= origins
    path = np.dot(rotation_matrix, path.transpose())

    for i in range(path.shape[1]):
        circle = plt.Circle((path[0, i], path[1, i]), radius=1, fill=True)
        ax.add_patch(circle)
        ax.arrow(path[0, i], path[1, i], np.cos(headings[i] + theta), np.sin(headings[i] + theta),
                 head_width=0.5, head_length=1.5, linewidth=0)

    # Show Grid Map ############################################################

    # Read grid map
    grid_map = mpimg.imread(image_file_name)
    # plt.imshow(grid_map)
    # plt.show()
    im_size = np.size(grid_map)
    resolution = 0.2
    im_width = grid_map.shape[0]
    im_height = grid_map.shape[1]
    pixels = grid_map.reshape(im_size)

    # Structure the 2d pixels as matrix
    u_coord = repmat(np.r_[0:im_width:1], im_height, 1).reshape(im_size)
    v_coord = repmat(np.c_[0:im_height:1], 1, im_width).reshape(im_size)

    # Find indexes of the obstacle pixels
    selected_indexes = where(0, pixels.tolist())

    # Trim coords
    u_coord = np.asarray(list(u_coord[selected_indexes])).transpose()
    v_coord = np.asarray(list(v_coord[selected_indexes])).transpose()

    # Transform the coords
    origins = repmat(np.r_[im_width / 2, im_height / 2], np.size(u_coord), 1).transpose()

    coords = (np.array([u_coord, v_coord]) - origins) * resolution
    plt.scatter(coords[0, :], coords[1, :], s=1, marker="s")

    # Show Products ############################################################

    my_list = read_csv(sproducts_file_name)

    for i in range(int(len(my_list) / 3)):
        coord = np.array([my_list[i * 3], my_list[i * 3 + 1]])
        origins = repmat(np.c_[origin], 1, coord.shape[1])
        coord = np.dot(rotation_matrix, coord - origins)
        plt.plot(coord[0, :], coord[1, :])
        fig.canvas.draw()
        input("next {}/{}?".format(i, int(len(my_list) / 3)))
    
    input("finish it?")


if __name__ == '__main__':
    main()
