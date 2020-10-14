import cv2
import os
import re
import json
import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image
from numpy import polyfit,poly1d
from scipy.optimize import leastsq
import numpy.linalg as LqA
from scipy.misc import derivative
path = "/home/gym/data/img/" #文件夹目录

l_c = 85
h_c = 160

def Ave_Color(vis,line):
    half = -1
    pix = vis.load()
    for x1,y1,x2,y2 in line:
        sum = np.array([0,0,0])
        for i in range(20):
            y = int(y1 + i/20*(y2-y1))
            x = max(4,int(x1 + i/20*(x2-x1)))
            sum += np.array(pix[x-3,y])
        sum = (sum/20).astype(int)
        if sum[0] < l_c and sum[1] < l_c and sum[2] < l_c:
            half = 0
        if sum[0] >h_c and sum[1] >h_c and sum[2] >h_c:
            half = 1
        if half == -1:
            return False
        sum = np.array([0,0,0])
        for i in range(20):
            y = int(y1 + i/20*(y2-y1))
            x = min(1276,int(x1 + i/20*(x2-x1)))
            sum += np.array(pix[x+3,y])
        sum = (sum/20).astype(int)
        if half == 1:
            if sum[0] < l_c and sum[1] < l_c and sum[2] < l_c:
                return True
            return False
        if half == 0:
            if sum[0] >h_c and sum[1] >h_c and sum[2] >h_c:
                return True
            return False

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
dirs= os.listdir(path) #得到文件夹下的所有文件名称
for dir in dirs: #遍历文件夹
    if not os.path.isdir(path+dir): #判断是否是文件夹，不是文件夹才打开
        continue
    #fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    #vout = cv2.VideoWriter(path+dir+'/lane_fit.avi',fourcc,30.0,(1280,720))
    cut_img_dir = path + dir +'/cut/'
    mask_folder = path + dir +'/mask/'
    cv_para_dir = path + dir +'/cv_para/'
    if not os.path.exists(mask_folder):
        continue
    dir = path + dir + '/para/'
    mkdir(cut_img_dir)
    mkdir(cv_para_dir)
    if not os.path.exists(dir):
        continue
    files = os.listdir(dir)
    files.sort(key = lambda x: int(x[:-5]))
    for file in files:
        if os.path.isdir(file):
            continue
        file_name = dir + file
        img_file_name = file_name.replace('para/','')[:-4]+'jpg'
        mask_img_name = file_name.replace('para/','mask/corrected_')[:-4]+'jpg'
        print("file mask_img_name {}".format(mask_img_name))
        mask_obj = cv2.imread(mask_img_name)
        mask_obj = cv2.cvtColor(mask_obj,cv2.COLOR_BGR2GRAY)
        vis = cv2.imread(img_file_name)
        vis_color = Image.open(img_file_name)

        masked_image = cv2.GaussianBlur(vis,(5,5),0,0)
        masked_image = cv2.cvtColor(masked_image,cv2.COLOR_BGR2GRAY)

        masked_image = cv2.Canny(masked_image,50,159)
        mask=np.zeros(masked_image.shape,dtype=np.uint8)
        roi_corners=np.array([[(0,360),(0,720),(1280,720),(1280,360)]],dtype=np.int32)
        ignore_mask_color = 255
        cv2.fillPoly(mask,roi_corners,ignore_mask_color)
        masked_image=cv2.bitwise_and(masked_image,mask)
        masked_image=cv2.bitwise_and(masked_image,mask_obj)
                ###### hough transformation
        rho = 1
        theta = np.pi/180
        threhold =15
        minlength = 20
        maxlengthgap = 20
        lines = cv2.HoughLinesP(masked_image,rho,theta,threhold,np.array([]),minlength,maxlengthgap)
#画线
        linecolor =[0,255,255]
        linewidth = 2
        masked_image = cv2.cvtColor(masked_image,cv2.COLOR_GRAY2BGR)
        push_data = []
        print("file name {}".format(file_name))
        if lines is not None:
            #print("lines length {}".format(len(lines)))
            for line in lines:
                for x1,y1,x2,y2 in line:
                    cv_para_data = {}
                    #print("line {},{},{},{}".format(x1,y1,x2,y2))
                    cv_para_data['endpoiont'] = line
                    cv_para_data['endpoiont'] = cv_para_data['endpoiont'].tolist()
                    #print("line type {}".format(type(line)))
                    cv_para_data['length'] = math.sqrt((y2-y1)**2+(x2-x1)**2)
                    cv_para_data['degree'] = math.degrees(math.atan2(y2-y1,x2-x1))%180
                    if cv_para_data['degree']<10 or cv_para_data['degree'] > 170:
                        continue
                    if not Ave_Color(vis_color,line):
                        continue
                    #print("cv_data_para {}".format(cv_para_data))
                    #print("cv_para_data {}".format(cv_para_data))
                    push_data.append(cv_para_data)
                    #print("push data {}".format(push_data))
                    cv2.line(masked_image,(x1,y1),(x2,y2),linecolor,linewidth)

        img_name = cut_img_dir + file[:-5]+".jpg"
        cv2.imwrite(img_name,masked_image)
        push_file_name = file_name.replace('para/','cv_para/')
        #print("push_data {}".format(push_data))
        with open(push_file_name,'w',encoding='utf-8') as push_file:
            json.dump(push_data,push_file,ensure_ascii=False)
        push_file.close()
