import cv2
import os
import re
import json
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from numpy import polyfit, poly1d
from scipy.optimize import leastsq
from scipy import stats
import numpy.linalg as LqA
from scipy.misc import derivative
import sympy as sp
import copy

path = "./challenge_testing_data/testing_data/video/img/283/"  # 文件夹目录


def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)


###### if want to use Polar Coords, see https://stackoom.com/question/3vV96/%E6%9B%B2%E7%BA%BF%E6%8B%9F%E5%90%88%E5%92%8Cmatplotlib. Here we just use x-y to y-x
# def PolarCoords(lane):
#    x_avg = x.mean()
#    y_avg = y.mean()
#    x = x - x_avg
#    y = y - y_avg
#    x = np.array(lane['x'])
#    y = np.array(lane['y'])
#    r = np.sqrt(x*x,y*y)
#    theta = np.degrees(arctan2(y,x))%360
#    return r,theta

###### lane-fit using leastsq with order 3

def Fun_3(p, x):
    a1, a2, a3, a4 = p
    return a1 * x ** 3 + a2 * x ** 2 + a3 * x + a4


def Error_3(p, x, y):
    return Fun_3(p, x) - y


def ShowLaneFit(x, y, x_fitted):
    plt.figure()
    plt.plot(x, y, 'r', label='Original curve')
    plt.plot(x_fitted, y, '-b', label='Fitted curve')
    plt.legend()
    plt.show()


def LaneFit_3(lane, vis=None):
    y = lane['y']
    x = lane['x']
    p_init = [0, 0, 0, 0.5]
    para = leastsq(Error_3, p_init, args=(y, x))
    x_fitted = Fun_3(para[0], y)
    for i in range(len(y)):
        p_ori = (int(x[i] * 1280), int(y[i] * 720))
        p_fit = (int(x_fitted[i] * 1280), int(y[i] * 720))
        cv2.circle(vis, p_ori, 5, (0, 255, 0), -1)
        cv2.circle(vis, p_fit, 5, (0, 0, 255), -1)
    # ShowLaneFit(x,y,x_fitted)
    return para


###### lane-fit using leastsq with order 6

def Fun_6(p, x):
    a1, a2, a3, a4, a5, a6, a7 = p
    return a1 * x ** 6 + a2 * x ** 5 + a3 * x ** 4 + a4 * x ** 3 + a5 * x ** 2 + a6 * x + a7


def Error_6(p, x, y):
    return Fun_6(p, x) - y


def LaneFit_6(lane):
    y = lane['y']
    x = lane['x']
    p_init = [0, 0, 0, 0, 0, 0, 0.5]
    para = leastsq(Error_6, p_init, args=(y, x))
    return para


###### evaluation

def tanh(x):
    s1 = np.exp(x) - np.exp(-x)
    s2 = np.exp(x) + np.exp(-x)
    s = s1 / s2
    return s


def Weight(lane):
    num = len(lane['x'])
    return tanh(num / 30.)


def Curve(para, lane):
    p = np.poly1d(para[0])
    C = []
    for y in lane['y']:
        d1 = np.polyder(p, 1)(y)
        d2 = np.polyder(p, 2)(y)
        c = abs(d2) / pow(1 + d1 * d1, 1.5)
        C.append(c)
    return C


def Evaluation(error, c, w):
    return w / (1280 * error + c)


def LaneFitMain(lane_data, file_name, vout=None):
    good_lane = []
    img_file_name = file_name.replace('line/', '')[:-4] + 'jpg'
    print("fitting img {}".format(img_file_name))
    vis = cv2.imread(img_file_name)
    vis_good_lane = cv2.imread(img_file_name)
    intercepts_3 = []
    ids = []
    scores = []
    sides = []
    for id in lane_data:
        if len(lane_data[id]['x']) < 7:  # Delete the detected line whose detected grids are less than 7
            continue
        lane_data[id]['x'] = np.array(lane_data[id]['x'])
        lane_data[id]['y'] = np.array(lane_data[id]['y'])
        para_3 = LaneFit_3(lane_data[id], vis)
        error = Error_3(para_3[0], lane_data[id]['y'], lane_data[id]['x'])
        error = np.std(error, ddof=1)
        # error = np.sqrt(np.sum(error**2))/len(error)
        # print("fitting {}th lane, with the total error {}.".format(id,error))
        para_6 = LaneFit_6(lane_data[id])
        w = Weight(lane_data[id])
        # print("weight {}".format(w))
        C = Curve(para_6, lane_data[id])
        c = np.std(C, ddof=1)
        # print("Curve std {}".format(c))
        score = Evaluation(error, c, w)
        intercept_3 = Fun_3(para_3[0], 1)
        intercept_6 = Fun_6(para_6[0], 1)
        print("{}th lane's score is {}; intercep3 is {}, intercep6 is {}".format(id, score, intercept_3, intercept_6))
        print("{}th lane's first grid is ({},{})".format(id, int(lane_data[id]['x'][0] * 1280),
                                                         int(lane_data[id]['y'][0] * 720)))
        if score > 0.8:
            LaneFit_3(lane_data[id], vis_good_lane)

        if -1 < intercept_3 < 2 and -1 < intercept_6 < 2 and abs(intercept_3 - intercept_6) < 0.05 and score > 0.3:
            intercepts_3.append(intercept_3)
            ids.append(id)
            scores.append(score)
            sides.append(left_or_right_side(para_3))

    # According to observation, here is an assumption that the inner line must be detected with as least as same high scores as that of the outer line.
    if sides.count('right') == 2:
        sides[sides.index('right') + 1] = 'outer_right'
    if sides.count('left') == 2:
        sides[sides.index('left')] = 'outer_left'
    for i in range(len(ids)):
        lane_detections['frame'].append(img_file_name.split('/')[-1])
        lane_detections['lane_id'].append(ids[i])
        lane_detections['score'].append(scores[i])
        lane_detections['intercept_3'].append(intercepts_3[i])
        lane_detections['label'].append(sides[i])

    vout.write(vis_good_lane)


def left_or_right_side(paras):  # classify which side of the screen side that the line is at
    bottom = Fun_3(paras[0], 1)
    middle = Fun_3(paras[0], 0.75)
    if bottom < middle:
        return 'left'
    else:
        return 'right'


dirs = os.listdir(path)  # 得到文件夹下的所有文件名称
lane_detections = {
    "frame": [],
    "lane_id": [],
    "score": [],
    "intercept_3": [],
    "label": []
}  # In label, cl, cr, sl, sr represents current_lane_left_line, current_lane_right_line, side_line_left_line, side_line_right_line

for dir in dirs:  # 遍历文件夹
    if not os.path.isdir(path + dir):  # 判断是否是文件夹，是文件夹才打开
        continue
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    vout = cv2.VideoWriter(path + dir + '/lane_fit.avi', fourcc, 30.0, (1280, 720))
    dir = path + dir
    # if os.path.exists(dir):
    #     continue
    # print('Hello')
    files = os.listdir(dir)
    files.sort(key=lambda x: int(x[:-5]))  # Nice!
    # print("files {}".format(files))
    for file in files:
        if os.path.isdir(file):
            continue
        file_name = dir + '/' + file
        f = open(file_name, 'r')
        pop_data = json.load(f)
        lane_data = {}
        for data in pop_data:
            lane_id = data["lane_id"]
            pos = data["pos"]
            pos[0] = pos[0] / 1280.
            pos[1] = pos[1] / 720.
            prob = data["prob"]
            if lane_id in lane_data.keys():
                lane_data[lane_id]['x'].append(pos[0])
                lane_data[lane_id]['y'].append(pos[1])
            else:
                lane_data[lane_id] = {}
                lane_data[lane_id]['x'] = [pos[0]]
                lane_data[lane_id]['y'] = [pos[1]]
            # print("lane_id: {}; pos: {}".format(lane_id, pos))
        LaneFitMain(lane_data, file_name, vout)
        f.close()
    vout.release()

# correct lines based on detected lines with high scores
# TODO: Whether optimize the line that has the best score.
# TODO: Consider lane changing #104.jpg
best_detection_idx = lane_detections['score'].index(max(lane_detections['score']))


def generate_lines_for_all_files(base_line, path, frame_id):
    lines = [0] * 150
    line = {}
    # Add the base line to the list lines
    line['frame'] = frame_id[:-4]
    line['x'] = base_line['x']
    line['y'] = base_line['y']
    lines[int(line['frame'][0])] = line
    line = {}
    idx_left = int(frame_id[:-4]) - 1
    idx_right = int(frame_id[:-4]) + 1

    # iterate from left and right side. Same y and check the difference between x.
    while idx_left >= 1:
        base_line_left = lines[idx_left + 1]
        data_json_file = path + str(idx_left) + '.json'
        f = open(data_json_file, 'r')
        pop_data = json.load(f)
        diff_x_y = {
            'x_diff': [],
            'y': []
        }
        for data in pop_data:
            if data['pos'][1] in base_line_left['y'] and abs(
                    data['pos'][0] - base_line_left['x'][base_line_left['y'].index(data['pos'][1])]) < 10:
                diff_x_y['x_diff'].append(
                    data['pos'][0] - base_line_left['x'][base_line_left['y'].index(data['pos'][1])])
                diff_x_y['y'].append(data['pos'][1])

        # correction
        line['frame'] = str(idx_left)
        if len(diff_x_y['y']) == 0:
            line['x'] = base_line_left['x']
        elif 0 < len(diff_x_y['y']) < 100:
            diff_x_pixel = sum(diff_x_y['x_diff']) / len(diff_x_y['x_diff'])
            line['x'] = [int(xxxx + diff_x_pixel) for xxxx in base_line_left['x']]
        else:
            slope, intercept, _, _, _ = stats.linregress(diff_x_y['y'], diff_x_y['x_diff'])
            line['x'] = [int(xxxx + (slope * yyyy + intercept)) for xxxx, yyyy in
                         zip(base_line_left['x'], base_line_left['y'])]
        line['y'] = base_line_left['y']
        lines[int(line['frame'])] = line
        line = {}
        idx_left -= 1

    while idx_right <= 149:
        base_line_right = lines[idx_right - 1]
        data_json_file = path + str(idx_right) + '.json'
        f = open(data_json_file, 'r')
        pop_data = json.load(f)
        diff_x = []
        diff_x_y = {
            'x_diff': [],
            'y': []
        }
        for data in pop_data:
            if data['pos'][1] in base_line_right['y'] and abs(
                    data['pos'][0] - base_line_right['x'][base_line_right['y'].index(data['pos'][1])]) < 3:
                # diff_x.append(data['pos'][0] - base_line_right['x'][base_line_right['y'].index(data['pos'][1])])
                diff_x_y['x_diff'].append(
                    data['pos'][0] - base_line_left['x'][base_line_left['y'].index(data['pos'][1])]) #TODO: Wherther use abs() here
                diff_x_y['y'].append(data['pos'][1])
        # correction
        line['frame'] = str(idx_right)
        if len(diff_x_y['y']) == 0:
            line['x'] = base_line_right['x']
        elif 0 < len(diff_x_y['y']) < 100:
            diff_x_pixel = sum(diff_x_y['x_diff']) / len(diff_x_y['x_diff'])
            line['x'] = [int(xxxx + diff_x_pixel) for xxxx in base_line_right['x']]
        else:
            slope, intercept, _, _, _ = stats.linregress(diff_x_y['y'], diff_x_y['x_diff'])
            line['x'] = [int(xxxx + (slope * yyyy + intercept)) for xxxx, yyyy in
                         zip(base_line_right['x'], base_line_right['y'])]

        line['y'] = base_line_right['y']
        lines[int(line['frame'])] = line
        line = {}
        idx_right += 1
    return lines


def find_max_score_in_list(score_list, indices):
    max = -1
    max_idx = -1
    for index in indices:
        if max < score_list[index]:
            max = score_list[index]
            max_idx = index
    return max, max_idx


if lane_detections['label'][best_detection_idx] == 'left':

    frame_id = lane_detections['frame'][best_detection_idx]
    lane_id = lane_detections['lane_id'][best_detection_idx]
    file_path = path + 'line/' + frame_id[:-3] + 'json'
    f = open(file_path, 'r')
    pop_data = json.load(f)
    base_left = {'x': [], 'y': []}
    for data in pop_data:
        if data['lane_id'] == lane_id:
            base_left['x'].append(data['pos'][0])
            base_left['y'].append(data['pos'][1])
    left_lines_for_all_files = generate_lines_for_all_files(base_left, path + 'line/', frame_id)
    another_line_indices = [i for i, x in enumerate(lane_detections['label']) if x == "right"]
    _, max_score_idx = find_max_score_in_list(lane_detections['score'], another_line_indices)
elif lane_detections['label'][best_detection_idx] == 'right':
    pass
else:
    pass

# testing

for line in left_lines_for_all_files:
    if line != 0:
        image_path = path + line['frame'] + '.jpg'
        img = cv2.imread(image_path)
        for x, y in zip(line['x'], line['y']):
            cv2.circle(img, (x, y), 5, (255, 0, 0))
        cv2.imwrite('corrected_img_{}.jpg'.format(int(line['frame'])), img)
        print(line['frame'])

a = 2
