import cv2
import os
import re
import json
import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from numpy import polyfit,poly1d
from scipy.optimize import leastsq
import numpy.linalg as LqA
from scipy.misc import derivative
path = "/home/gym/video/img/" #文件夹目录

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)

def Fun_3(p,x):
    a1,a2,a3,a4 = p
    return a1*x**3 + a2*x**2 + a3*x +a4

dirs= os.listdir(path) #得到文件夹下的所有文件名称
for dir in dirs: #遍历文件夹
    if not os.path.isdir(path+dir): #判断是否是文件夹，不是文件夹才打开
        continue
    mask_folder = path+dir+'/mask_folder/'
    if not os.path.exists(mask_folder):
        continue
    ML_para_dir = path + dir + '/para/'
    if not os.path.exists(ML_para_dir):
        continue
    CV_para_dir = path + dir + '/cv_para/'
    if not os.path.exists(CV_para_dir):
        continue
    files = os.listdir(ML_para_dir)
    files.sort(key = lambda x: int(x[:-5]))
    for file in files:
        if os.path.isdir(file):
            continue
        file_name = ML_para_dir + file
        f = open(file_name,'r')
        cv_para_file_name = file_name.replace('para/','cv_para/')
        cv_f = open(cv_para_file_name,'r')
        mask_img_name = file_name.replace('para/','mask_folder/')[:-4]+'jpg'
        mask_obj = cv2.imread(mask_img_name)
        maks_obj=cv2.cvtColor(mask_obj, cv2.COLOR_BGR2GRAY)
        pop_data = json.load(f)
        cv_pop_data = json.load(cv_f)
        lane_data = {}
        for data in pop_data:
            for lane_id in data:
                para_3 = np.array(data[lane_id]["para_3"])
                if len(para_3) == 0:
                    print("file name {}".format(file_name))
                    print("NO FIT PARA HERE")
                    continue
                best_line = np.array([])
                best_line_length = 0.
                for cv_data in cv_pop_data:
                    line = np.array(cv_data["endpoiont"][0])
                    x1,y1,x2,y2 = line
                    #print("line {},{},{},{}".format(x1,y1,x2,y2))
                    degree = cv_data["degree"]
                    x1_fit = Fun_3(para_3,y1/720.)*1280.
                    x2_fit = Fun_3(para_3,y2/720.)*1280.
                    degree_fit = math.degrees(math.atan2(y2-y1,x2_fit-x1_fit))%180
                    x_mid_fit = Fun_3(para_3,0.5*(y1+y2)/720.)*1280.
                    x_mid = 0.5*(x1+x2)
                    if abs(x_mid_fit-x_mid) > 30 or abs(degree_fit-degree) > 30:
                        #print("BAD FIT LANE")
                        continue
                    if cv_data["length"]>best_line_length:
                        best_line_length = cv_data['length']
                        best_line = line
                    #print("x_mid_fit {} x_mid {}".format(x_mid_fit,x_mid))
                    #print("degree_fit {} degree {}".format(degree_fit,degree))
                    #print("degree diff {}".format(degree_fit-degree))
                    #print("mid diff {}".format(x_mid_fit-x_mid))

                if len(best_line) == 0:
                    print("NO GOOD FIT LANE FOR THIS AREA")
                    continue
                mask_area = 0
                y_min = min(best_line[1],best_line[3])-100
                y_max = max(best_line[1],best_line[3])
                x_min = min(best_line[0],best_line[2])
                x_max = max(best_line[0],best_line[2])
                for i in range(y_min,y_max):
                    for j in range(x_min,x_max):
                        if mask_obj[i][j][0] > 200:
                            mask_area += 1.
                cover_part = mask_area/(x_max-x_min)
                if y_max-y_min-cover_part > 40:
                    print("solild line")
                else:
                    print("dotted line")
        f.close()
        cv_f.close()







