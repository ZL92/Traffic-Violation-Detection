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
path = "./challenge_testing_data/testing_data/video1/img/" #文件夹目录

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)

def last_flag(data,last_frame_data):
    for d in last_frame_data:
        if abs(data["x_when_y_650"] - d[0]) < 50./1280:
            # print("works")
            # print(last_frame_data)
            return d[1]
    return data["solid"]

def Fun_3(p,x):
    a1,a2,a3,a4 = p
    return a1*x**3 + a2*x**2 + a3*x +a4

dirs= os.listdir(path) #得到文件夹下的所有文件名称
for dir in dirs: #遍历文件夹
    print('In solid_dot_lane.py: ', dir)
    # if path+dir !=path+ "52":
    #     continue
    if not os.path.isdir(path+dir): #判断是否是文件夹，不是文件夹才打开
        continue
    mask_folder = path+dir+'/mask/'
    if not os.path.exists(mask_folder):
        continue
    Final_para_dir = path + dir + '/corrected_lines_data/'
    if not os.path.exists(Final_para_dir):
        continue
    CV_para_dir = path + dir + '/cv_para/'
    if not os.path.exists(CV_para_dir):
        continue
    mkdir(path + dir + '/corrected_lines_data_final/')
    files = os.listdir(Final_para_dir)
    files.sort(key = lambda x: int(x[:-5]))
    last_frame_data = []
    for file in files:
        # print("file {}".format(file))
        if os.path.isdir(file):
            continue
        file_name = Final_para_dir + file
        f = open(file_name,'r')
        cv_para_file_name = file_name.replace('corrected_lines_data/','cv_para/')
        cv_f = open(cv_para_file_name,'r')
        mask_img_name = file_name.replace('corrected_lines_data/','mask/corrected_')[:-4]+'jpg'
        print("mask_img_name {}".format(mask_img_name))
        mask_obj = cv2.imread(mask_img_name)
        maks_obj=cv2.cvtColor(mask_obj, cv2.COLOR_BGR2GRAY)
        pop_data = json.load(f)
        cv_pop_data = json.load(cv_f)
        lane_data = []
        this_frame_data = []
        for data in pop_data:
            para_3 = np.array(data["para_3"])
            if len(para_3) == 0:
                print("NO FIT PARA HERE")
                continue
            best_line = np.array([])
            best_line_length = 0.
            x1_fit = Fun_3(para_3,600/720.)*1280.
            x2_fit = Fun_3(para_3,700/720.)*1280.
            degree_fit = math.degrees(math.atan2(100,x2_fit-x1_fit))%180

            for cv_data in cv_pop_data:
                line = np.array(cv_data["endpoiont"][0])
                x1,y1,x2,y2 = line
                degree = cv_data["degree"]

                x_mid_fit = Fun_3(para_3,0.5*(y1+y2)/720.)*1280.
                x_mid = 0.5*(x1+x2)
                if abs(x_mid_fit-x_mid) > 30 or abs(degree_fit-degree) > 15:
                    # print("BAD FIT LANE")
                    continue
                if cv_data["length"]>best_line_length:
                    best_line_length = cv_data['length']
                    best_line = line
                    # print("best_line_length {} ".format(best_line_length))
                #print("x_mid_fit {} x_mid {}".format(x_mid_fit,x_mid))
                #print("degree_fit {} degree {}".format(degree_fit,degree))
                #print("degree diff {}".format(degree_fit-degree))
                #print("mid diff {}".format(x_mid_fit-x_mid))

            if len(best_line) == 0:
                data["solid"] = last_flag(data,last_frame_data)
                lane_data.append(data)
                this_frame_data.append((data["x_when_y_650"],data["solid"]))
                print("NO GOOD FIT LANE FOR THIS AREA")
                continue
            x_mask_area = 0
            y_mask_area = 0
            _x,_y = 0,0
            y_min = min(best_line[1],best_line[3])
            y_max = max(best_line[1],best_line[3])
            x_min = min(best_line[0],best_line[2])
            x_max = max(best_line[0],best_line[2])
            if y_min == best_line[1]:
                _x,_y = best_line[0],best_line[1]
            else:
                _x,_y = best_line[2],best_line[3]
            # print(best_line)
            # print(_x,_y)
            if y_max < 680:
                # print("too high")
                data["solid"] = False
            elif (y_max - y_min) > 300:
                # print("too long")
                data["solid"] = True
            else:
                # for i in range(max(0,_y-5),min(720,_y+5)):
                #     for j in range(max(0,_x-30),min(1280,_x+30)):
                for i in range(_y-5,_y+5):
                    for j in range(_x-30,_x+30):
                        if mask_obj[i][j][0] < 200:
                            x_mask_area += 1.
                x_cover_part = x_mask_area/10
                if x_cover_part > 20:
                    # print("x_cover_part {}".format(x_cover_part))
                    data["solid"] = last_flag(data,last_frame_data)
                else:
                    # print("not x_cover_part")
                    for i in range(_y-50,_y):
                        for j in range(_x-5,_x+5):
                            if mask_obj[i][j][0] < 200:
                                y_mask_area += 1.
                    y_cover_part = y_mask_area/10
                    if y_cover_part > 35:
                        # print("y_cover_part")
                        data["solid"] = last_flag(data,last_frame_data)
                    else:
                        # print("not y_cover_part")
                        data["solid"] = False
            lane_data.append(data)
            this_frame_data.append((data["x_when_y_650"],data["solid"]))
        last_frame_data = this_frame_data
        push_file_name = file_name.replace('corrected_lines_data/','corrected_lines_data_final/')
        with open(push_file_name,'w',encoding='utf-8') as push_file:
            json.dump(lane_data,push_file,ensure_ascii=False)
        f.close()
        cv_f.close()