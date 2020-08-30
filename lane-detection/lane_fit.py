import cv2
import os
import re
import json
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from numpy import polyfit,poly1d
from scipy.optimize import leastsq
path = "/home/gym/video/img/" #文件夹目录
def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)

###### if want to use Polar Coords, see https://stackoom.com/question/3vV96/%E6%9B%B2%E7%BA%BF%E6%8B%9F%E5%90%88%E5%92%8Cmatplotlib. Here we just use x-y to y-x
#def PolarCoords(lane):
#    x_avg = x.mean()
#    y_avg = y.mean()
#    x = x - x_avg
#    y = y - y_avg
#    x = np.array(lane['x'])
#    y = np.array(lane['y'])
#    r = np.sqrt(x*x,y*y)
#    theta = np.degrees(arctan2(y,x))%360
#    return r,theta

###### lane-fit using leastsq

def Fun(p,x):
    a1,a2,a3 = p
    return a1*x**2 + a2*x + a3

def Error(p,x,y):
        return Fun(p,x) - y

def ShowLaneFit(x,y,x_fitted):
    plt.figure()
    plt.plot(x,y,'r', label = 'Original curve')
    plt.plot(x_fitted,y,'-b', label ='Fitted curve')
    plt.legend()
    plt.show()

def LaneFit(lane,vis=None):
    y = lane['y']
    x = lane['x']
    p_init = [0,0,0.5]
    para = leastsq(Error,p_init,args=(y,x))
    x_fitted = Fun(para[0],y)
    for i in range(len(y)):
        p_ori = (int(x[i]*1280),int(y[i]*720))
        p_fit = (int(x_fitted[i]*1280),int(y[i]*720))
        cv2.circle(vis,p_ori,5,(0,255,0),-1)
        cv2.circle(vis,p_fit,5,(0,0,255),-1)

    #print(para[0])
    #ShowLaneFit(x,y,x_fitted)
    return para

def LaneFitMain(lane_data,file_name,vout=None):
    img_file_name = file_name.replace('line/','')[:-4]+'jpg'
    print("fitting img {}".format(img_file_name))
    vis = cv2.imread(img_file_name)
    for id in lane_data:
        if len(lane_data[id]['x']) < 10:
            continue
        lane_data[id]['x'] = np.array(lane_data[id]['x'])
        lane_data[id]['y'] = np.array(lane_data[id]['y'])
        para = LaneFit(lane_data[id],vis)
        error = Error(para[0],lane_data[id]['y'],lane_data[id]['x'])
        error = np.sqrt(np.sum(error**2))
        print("fitting {}th lane, with the total error {}.".format(id,error))
    vout.write(vis)

dirs= os.listdir(path) #得到文件夹下的所有文件名称
for dir in dirs: #遍历文件夹
    if not os.path.isdir(path+dir): #判断是否是文件夹，不是文件夹才打开
        continue
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    vout = cv2.VideoWriter(path+dir+'/lane_fit.avi',fourcc,30.0,(1280,720))
    dir = path + dir + '/line/'
    if not os.path.exists(dir):
        continue
    files = os.listdir(dir)
    files.sort(key = lambda x: int(x[:-5]))
    print("files {}".format(files))
    for file in files:
        if os.path.isdir(file):
            continue
        file_name = dir + file
        f = open(file_name,'r')
        pop_data = json.load(f)
        lane_data = {}
        for data in pop_data:
            lane_id = data["lane_id"]
            pos = data["pos"]
            pos[0] = pos[0]/1280.
            pos[1] = pos[1]/720.
            prob = data["prob"]
            if lane_id in lane_data.keys():
                lane_data[lane_id]['x'].append(pos[0])
                lane_data[lane_id]['y'].append(pos[1])
            else:
                lane_data[lane_id] = {}
                lane_data[lane_id]['x']=[pos[0]]
                lane_data[lane_id]['y']=[pos[1]]
            #print("{}  {}".format(lane_id,pos))
        LaneFitMain(lane_data,file_name,vout)
        f.close()
    vout.release()







