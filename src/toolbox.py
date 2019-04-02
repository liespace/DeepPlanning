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


def write_csv(my_list, file_name, delimiter=','):
    with open(file_name, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=delimiter, quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for item in my_list:
            csv_writer.writerow(item)
