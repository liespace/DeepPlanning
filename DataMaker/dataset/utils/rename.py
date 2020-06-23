#!/usr/bin/env python
import os

path='./blue'    
fileList=os.listdir(path)

for name in fileList:
    oldpath = path + os.sep + name
    print(oldpath)
    parts = name.split('_')
    number = int(parts[0]) + 8700
    if len(parts) == 3:
        new_name = str(number) + '_' + parts[1] + '_' + parts[2]
    else:
        new_name = str(number) + '_' + parts[1]
    newpath = path + os.sep + new_name
    print(newpath)
    os.rename(oldpath,newpath)