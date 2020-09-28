import cv2
import os
import re
import json
import math
import random
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from numpy import polyfit,poly1d
from scipy.optimize import leastsq
import numpy.linalg as LqA
from scipy.misc import derivative
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import matplotlib.pyplot as plt
path = "/home/gym/video/img/" #文件夹目录

image_x = 1280
image_y = 720

def show(x,y):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_title('Scatter Plot')
    plt.xlabel('X')
    plt.ylabel('Y')
    ax1.scatter(x,y,c = 'r',marker = 'o')
    plt.legend('x1')
    plt.show()

def Fun_3(p, y):
    a1, a2, a3, a4 = p
    return a1 * y ** 3 + a2 * y ** 2 + a3 * y + a4

def Linear(x1,x2):
    diff = x1 - x2
    diff[abs(diff)<0.002] = 10*diff[abs(diff)<0.002]
    diff[abs(diff)<0.02] = 0.1*diff[abs(diff)<0.02]
    return diff

def Error_3_1(p, y, x):
    return Linear(Fun_3(p, y),x)

def Error_3(p, y, x):
    return Fun_3(p, y)-x

def LaneFit_3(y, x):
    p_init = [0, 0, 0, 0.5]
    para = leastsq(Error_3, p_init, args=(y, x))
    return para

def Fun_6(p, y):
    a1, a2, a3, a4, a5, a6, a7 = p
    return a1 * y ** 6 + a2 * y ** 5 + a3 * y ** 4 + a4 * y ** 3 + a5 * y ** 2 + a6 * y + a7

def Error_6(p, y, x):
    return Fun_6(p, y)-x

def LaneFit_6(y, x):
    p_init = [0, 0, 0, 0, 0, 0, 0.5]
    para = leastsq(Error_6, p_init, args=(y, x))
    return para

def fit_1st_quartile(fit_data_list):
    scores = []
    for data in fit_data_list:
        scores.append(data['score'])
    return np.percentile(scores, 80)

def tanh(x):
    s1 = np.exp(x) - np.exp(-x)
    s2 = np.exp(x) + np.exp(-x)
    s = s1 / s2
    return s

def Weight(x):
    num = len(x)
    return tanh(num / 30.)

def Curve(para, Y):
    p = np.poly1d(para[0])
    C = []
    for y in Y:
        d1 = np.polyder(p, 1)(y)
        d2 = np.polyder(p, 2)(y)
        c = abs(d2) / pow(1 + d1 * d1, 1.5)
        C.append(c)
    return C

def Evaluation(error, c, w):
    return w / (1280 * error + c)

def grid_cv(endpoints,fa,fb):
    '''
    :param endpoints: cv result
    :return: scaled in between 0-1
    '''
    # regression line
    x1, y1, x2, y2 = endpoints
    grid = {
        'x': [],
        'y': []
    }
    a = (x2 - x1) / (y2 - y1)
    b = x1 - a * y1
    gridding_num = 20  # Setting is in configs/tusimple.py
    col_sample = np.linspace(y1, y2, gridding_num)
    for col in col_sample:
        x_hat = a * col + b
        x_fit = fa * col + fb
        grid['x'].append((0.2*x_hat + 0.8*x_fit)/image_x)
        grid['y'].append(col / image_y)
    return grid

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
dirs= os.listdir(path) #得到文件夹下的所有文件名称
for dir in dirs: #遍历文件夹
    final_frame_lines = []
    if not os.path.isdir(path+dir): #判断是否是文件夹，不是文件夹才打开
        continue
    if dir != "283":
        continue
    #fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    #vout = cv2.VideoWriter(path+dir+'/lane_fit.avi',fourcc,30.0,(1280,720))
    cv_para_dir = path + dir +'/cv_para/'
    dir = path + dir + '/para/'
    if not os.path.exists(cv_para_dir):
        continue
    if not os.path.exists(dir):
        continue
    files = os.listdir(dir)
    files.sort(key = lambda x: int(x[:-5]))
    for file in files:
        if os.path.isdir(file):
            continue
        fit_para_path = dir + file
        cv_para_path = fit_para_path.replace('para','cv_para')
        # img_file_name = file_name.replace('para/','')[:-5]+'jpg'
        # vis = cv2.imread(img_file_name)
        try: # some video has less frames than 150
            f_cv = open(cv_para_path, 'r')
            cv_data = json.load(f_cv)
        except:
            print('failed to open {}'.format(cv_para_path))
            continue
        try:
            f_fit = open(fit_para_path, 'r')
            fit_data = json.load(f_fit)[0]
        except:
            print('failed to open {}'.format(fit_para_path))
            continue
        # print("fit_data.value {}".format(fit_data.values()))
        for line_data in fit_data.values():
            if len(line_data['para_3']) == 0:
                continue
            final_line_data = {}
            if line_data['score'] > 0.8:
                final_line_data['frame'] = int(file[:-5])
                final_line_data['x']=line_data['x']
                final_line_data['y']=line_data['y']
                final_line_data['para_3'] = para_3
                final_line_data['x_when_y_650'] = Fun_3(para_3[0], 650/image_y)
                final_line_data['solid'] = True
                final_line_data['overall_score'] = line_data['score']
                final_line_data['inferring'] = 'inferred' # left, right, inferred
                final_frame_lines.append(final_line_data)
                # final_frame_lines['frame'] = fit_data['frame']
                # final_frame_lines['line_data'] = final_line_data
                # video_all_lines.append(final_frame_lines)

            elif line_data['score'] > 0.6:
                final_line_data['frame'] = int(file[:-5])
                fx1, fx2 = line_data['x'][0] * image_x, line_data['x'][-1] * image_x
                fy1, fy2 = line_data['y'][0] * image_y, line_data['y'][-1] * image_y
                fa = (fx2 - fx1) / (fy2 - fy1)
                fb = fx1 - fa * fy1
                margin = 30
                polygon = Polygon([(fx1 - margin, fy1), (fx1 + margin, fy1), (fx2 + margin, fy2), (fx2 - margin, fy2)])
                fit_degree1 = math.degrees(math.atan2(fy2-fy1,fx2-fx1+2*margin))%180
                fit_degree2 = math.degrees(math.atan2(fy2-fy1,fx2-fx1-2*margin))%180
                degree_min = min(fit_degree1,fit_degree2)
                degree_max = max(fit_degree1,fit_degree2)
                matched_cv_grid_list = []
                for cv_signle_data in cv_data:
                    x1, y1, x2, y2 = cv_signle_data['endpoiont'][0]
                    # check whether the middle point of the cv data is in the region
                    point_cv = Point((x1 + x2) / 2, (y1 + y2) / 2)
                    # print(polygon.contains(point_cv))
                    if polygon.contains(point_cv) and cv_signle_data['length'] > 20 \
                    and cv_signle_data['degree'] > degree_min and cv_signle_data['degree'] < degree_max:
                        cv_line_data = cv_signle_data['endpoiont'][0]
                        cv_grid_data = grid_cv(cv_line_data,fa,fb)
                        matched_cv_grid_list.append(cv_grid_data)
                if len(matched_cv_grid_list) == 0:
                    continue
                num_fit = len(line_data['x'])
                num_cv_grid = sum([len(cv['x']) for cv in matched_cv_grid_list])
                cv_vs_fit_ratio = 0.5  # selected cv is 3 times of fit data
                # print("num_fit {} num_cv_grid {} ".format(num_fit,num_cv_grid))
                # Resample from cv and fit data
                cv_grid_x = [x['x'] for x in matched_cv_grid_list]
                cv_grid_y = [x['y'] for x in matched_cv_grid_list]
                # print("cv_grid_x {} cv_grid_y {} ".format(cv_grid_x,cv_grid_y))
                flattened_cv_x = [y for x in cv_grid_x for y in x]
                flattened_cv_y = [y for x in cv_grid_y for y in x]
                pairs_cv = list(zip(flattened_cv_x, flattened_cv_y))
                # print("pairs_cv {}".format(pairs_cv))
                pairs_cv = random.sample(pairs_cv, int(num_fit * cv_vs_fit_ratio))
                fit_x, fit_y = line_data['x'], line_data['y']
                cv_x, cv_y = [x[0] for x in pairs_cv], [x[1] for x in pairs_cv]
                resampled_x = fit_x + cv_x
                resampled_y = fit_y + cv_y
                resampled_x = np.asarray(resampled_x)
                resampled_y = np.asarray(resampled_y)
                # show(cv_x,cv_y)
                if len(resampled_x)<=7:
                    continue
                para_3 = LaneFit_3(resampled_y, resampled_x)
                error = Error_3(para_3[0], resampled_y, resampled_x)
                error = np.std(error, ddof=1)
                w = Weight(resampled_x)
                v2 = 2*abs(para_3[0][0])+abs(para_3[0][1])
                score = Evaluation(error/2, v2, w)

                if score>0.8:
                    final_line_data['x']=resampled_x
                    final_line_data['y']=resampled_y
                    final_line_data['para_3'] = para_3
                    final_line_data['x_when_y_650'] = Fun_3(para_3[0], 650/image_y)
                    final_line_data['solid'] = True
                    final_line_data['overall_score'] = score
                    final_line_data['inferring'] = 'inferred' # left, right, inferred
                    final_frame_lines.append(final_line_data)
                    # final_frame_lines['frame'] = fit_data['frame']
                    # final_frame_lines['line_data'] = final_line_data
                    # video_all_lines.append(final_frame_lines)
        # print("final_frame_lines {}".format(final_frame_lines))

        #print("push_data {}".format(push_data))
        # with open(push_file_name,'w',encoding='utf-8') as push_file:
        #     json.dump(push_data,push_file,ensure_ascii=False)
        # push_file.close()
