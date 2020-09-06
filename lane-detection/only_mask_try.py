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
    #fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    #vout = cv2.VideoWriter(path+dir+'/lane_fit.avi',fourcc,30.0,(1280,720))
    cut_img_dir = path + dir +'/cut/'
    dir = path + dir + '/para/'
    mkdir(cut_img_dir)
    if not os.path.exists(dir):
        continue
    files = os.listdir(dir)
    files.sort(key = lambda x: int(x[:-5]))
    for file in files:
        if os.path.isdir(file):
            continue
        file_name = dir + file
        f = open(file_name,'r')
        img_file_name = file_name.replace('para/','')[:-4]+'jpg'
        mask_img_name = file_name.replace('para/','mask_folder/')[:-4]+'jpg'
        mask_obj = cv2.imread(mask_img_name)
        mask_obj = cv2.cvtColor(mask_obj,cv2.COLOR_BGR2GRAY)
        vis = cv2.imread(img_file_name)
        pop_data = json.load(f)
        lane_data = {}
        for data in pop_data:
            for lane_id in data:
                para = data[lane_id]["para"]
                if len(para) == 0:
                    continue
                y_max = max(data[lane_id]['y'])
                y_min = min(data[lane_id]['y'])
                y_upper = int(720 * y_min)
                y_lower = int(720 * y_max)
                x_upper_left = int(1280*min(1.,max(Fun_3(para,y_min)-0.05,0.0)))
                x_upper_right = int(1280*max(0., min(Fun_3(para,y_min)+0.05,1.0)))
                x_lower_left = int(1280*min(1.,max(Fun_3(para,y_max)-0.05,0.0)))
                x_lower_right = int(1280*max(0.,min(Fun_3(para,y_max)+0.05,1.0)))
                print("file:{} id:{} ({:.0f},{:.0f},{:.0f}) to ({:.0f},{:.0f},{:.0f})".format(file,lane_id,x_upper_left,x_upper_right,y_upper,x_lower_left,x_lower_right,y_lower))
                masked_image = cv2.GaussianBlur(vis,(5,5),0,0)
                masked_image = cv2.cvtColor(masked_image,cv2.COLOR_BGR2GRAY)
                masked_image = cv2.Canny(masked_image,50,159)
                mask=np.zeros(masked_image.shape,dtype=np.uint8)
                #roi_corners=np.array([[(x_upper_left,y_upper),(x_upper_right,y_upper),(x_lower_right,y_lower),(x_lower_left,y_lower)]],dtype=np.int32)
                roi_corners=np.array([[(320,360),(0,720),(1280,720),(960,360)]],dtype=np.int32)
                ignore_mask_color = 255
                cv2.fillPoly(mask,roi_corners,ignore_mask_color)
                masked_image=cv2.bitwise_and(masked_image,mask)
                masked_image=cv2.bitwise_and(masked_image,mask_obj)
                ###### hough transformation
                rho = 1
                theta = np.pi/180
                threhold =15
                minlength = 40
                maxlengthgap = 20
                lines = cv2.HoughLinesP(masked_image,rho,theta,threhold,np.array([]),minlength,maxlengthgap)
#画线
                linecolor =[0,255,255]
                linewidth = 2
                masked_image = cv2.cvtColor(masked_image,cv2.COLOR_GRAY2BGR)
                if lines is not None:
                    degree1 = math.degrees(math.atan2(y_upper-y_lower,x_upper_right-x_lower_left))%180
                    degree2 = math.degrees(math.atan2(y_upper-y_lower,x_upper_left-x_lower_right))%180
                    #print("degree {},{}".format(degree1,degree2))
                    for line in lines:
                        for x1,y1,x2,y2 in line:
                            degree = math.degrees(math.atan2(y2-y1,x2-x1))%180
                            if degree > min(degree1,degree2) and degree < max(degree1,degree2):
                                cv2.line(masked_image,(x1,y1),(x2,y2),linecolor,linewidth)
                            #cv2.line(masked_image,(x1,y1),(x2,y2),linecolor,linewidth)
                '''
                for i in range(len(data[lane_id]['y'])):
                    p_ori = (int(data[lane_id]['x'][i]*1280),int(data[lane_id]['y'][i]*720))
                    cv2.circle(masked_image,p_ori,5,(0,255,0),-1)
                '''
                img_name = cut_img_dir + file[:-4]+'_'+str(lane_id)+".jpg"
                cv2.imwrite(img_name,masked_image)
        f.close()
        #vout.write(vis)
    #vout.release()







