import csv


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
