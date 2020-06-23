#!/usr/bin/env python
import os
from shutil import move, copy

wellpath = './well'    
rarepath = './rare'
donepath = './done'
fileList=os.listdir(wellpath)

for name in fileList:
    number = name.split('_')[0]
    rare = number + '_encoded.png'
    frarepath = rarepath + os.sep + rare
    trarepath = donepath + os.sep + rare
    fwellpath = wellpath + os.sep + name
    twellpath = donepath + os.sep + name
    print(frarepath + '-->>' + trarepath)
    print(fwellpath + '-->>' + twellpath)
    copy(frarepath, trarepath)
    copy(fwellpath, twellpath)