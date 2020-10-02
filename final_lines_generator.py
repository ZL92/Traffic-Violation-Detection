import cv2
import os
import re
import json
import math
import random
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from numpy import polyfit, poly1d
from scipy.optimize import leastsq
import numpy.linalg as LqA
from scipy.misc import derivative
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import matplotlib.pyplot as plt
import copy
import time
random.seed(30)

# path = "/home/gym/video/img/" #文件夹目录
path = "./challenge_testing_data/testing_data/video/img/"

image_x = 1280
image_y = 720


def show(x, y, frame=-1):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    if frame != -1:
        ax1.set_title('Scatter Plot frame{}'.format(frame))
    else:
        ax1.set_title('Scatter Plot')
    plt.xlabel('X')
    plt.ylabel('Y')
    ax1.scatter(x, y, c='r', marker='o')
    plt.legend('x1')
    plt.show()


def Fun_3(p, y):
    a1, a2, a3, a4 = p
    return a1 * y ** 3 + a2 * y ** 2 + a3 * y + a4


def Linear(x1, x2):
    diff = x1 - x2
    diff[abs(diff) < 0.002] = 10 * diff[abs(diff) < 0.002]
    diff[abs(diff) < 0.02] = 0.1 * diff[abs(diff) < 0.02]
    return diff


def Error_3_1(p, y, x):
    return Linear(Fun_3(p, y), x)


def Error_3(p, y, x):
    return Fun_3(p, y) - x


def LaneFit_3(y, x):
    p_init = [0, 0, 0, 0.5]
    para = leastsq(Error_3, p_init, args=(y, x))
    return para


def Fun_6(p, y):
    a1, a2, a3, a4, a5, a6, a7 = p
    return a1 * y ** 6 + a2 * y ** 5 + a3 * y ** 4 + a4 * y ** 3 + a5 * y ** 2 + a6 * y + a7


def Error_6(p, y, x):
    return Fun_6(p, y) - x


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


def grid_cv(endpoints, fa, fb):
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
        grid['x'].append((0.2 * x_hat + 0.8 * x_fit) / image_x)
        grid['y'].append(col / image_y)
    return grid


def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)


def infer_line(origin_line, infer_label):
    '''
    Assign infered line score 0.6
    :param origin_line:
    :param infer_label: 'left' or 'right'
    :return:
    '''

    new_line = {
        'frame': [],
        'x': [],
        'y': [],
        'para_3': [0, 0, 0, 0],
        'x_when_y_650': [],
        'solid': True,  # Ture or False
        'overall_score': 0.6,
        'inferring': infer_label  # from_left, from_right, inferred
    }

    fx1, fx2 = origin_line['x'][0] * image_x, origin_line['x'][-1] * image_x
    fy1, fy2 = origin_line['y'][0] * image_y, origin_line['y'][-1] * image_y
    fa = (fx2 - fx1) / (fy2 - fy1)
    fb = fx1 - fa * fy1
    margin = 20
    polygon = Polygon([(fx1 - margin, fy1), (fx1 + margin, fy1), (fx2 + margin, fy2), (fx2 - margin, fy2)])
    fit_degree1 = math.degrees(math.atan2(fy2 - fy1, fx2 - fx1 + 2 * margin)) % 180
    fit_degree2 = math.degrees(math.atan2(fy2 - fy1, fx2 - fx1 - 2 * margin)) % 180
    degree_min = min(fit_degree1, fit_degree2)
    degree_max = max(fit_degree1, fit_degree2)

    #   load cv
    if infer_label == 'from_left':
        infer_line_frame = origin_line['frame'] + 1
    else:
        infer_line_frame = origin_line['frame'] - 1
    f_cv = open(cv_para_dir + str(infer_line_frame) + '.json', 'r')
    cv_data = json.load(f_cv)

    # calculate closed cv result and grid it
    matched_cv_grid_list = []
    for cv_signle_data in cv_data:
        x1, y1, x2, y2 = cv_signle_data['endpoiont'][0]
        # check whether the middle point of the cv data is in the region
        point_cv = Point((x1 + x2) / 2, (y1 + y2) / 2)
        # print(polygon.contains(point_cv))
        if polygon.contains(point_cv) and cv_signle_data['length'] > 20 \
                and cv_signle_data['degree'] > degree_min and cv_signle_data['degree'] < degree_max:
            cv_line_data = cv_signle_data['endpoiont'][0]
            cv_grid_data = grid_cv(cv_line_data, fa, fb)
            matched_cv_grid_list.append(cv_grid_data)

    #   Select close cv data and calculate difference
    diff_x_y = {
        'x_diff': [],
        'y': []
    }

    allowed_error_y = 2 / 1280  # As cv and origin data are in different grid, this is used for selecting the cv data that has the 'same' y value to that of origin data
    for x, y in zip(origin_line['x'], origin_line['y']):
        for idx in range(len(matched_cv_grid_list)):
            for x_cv, y_cv in zip(matched_cv_grid_list[idx]['x'], matched_cv_grid_list[idx]['y']):
                if abs(y - y_cv) <= allowed_error_y and abs(
                        x_cv - x) < 20 / 1280:  # To get rid of the parallel cv result
                    diff_x_y['x_diff'].append(x_cv - x)
                    diff_x_y['y'].append(y)

    #   Correction
    new_line['frame'] = infer_line_frame
    if len(diff_x_y['y']) == 0:
        new_line['x'] = origin_line['x']
        new_line['y'] = origin_line['y']
        new_line['x_when_y_650'] = origin_line['x_when_y_650']
    else:
        diff_x_pixel = sum(diff_x_y['x_diff']) / len(diff_x_y['x_diff'])
        new_line['x'] = [(xxxx + diff_x_pixel) for xxxx in origin_line['x']]
        new_line['y'] = origin_line['y']
        new_line['x_when_y_650'] = origin_line['x_when_y_650'] + diff_x_pixel
    return new_line


def clean_conflict_line(final_frame_lines, print_cleaning=True):
    '''
    Check whether two results of the same frame have closed values of 'x_when_y_659' using threshold 100.
    If detected, remove the one with the lower score and change 'inferring' of the higher one to 'inferred'
    '''
    cleaned = 0
    for result1 in final_frame_lines:
        for result2 in final_frame_lines:
            if result1['x_when_y_650'] != result2['x_when_y_650'] and result1['frame'] == result2['frame']:
                if abs(result1['x_when_y_650'] - result2['x_when_y_650']) < (100 / 1280):
                    cleaned += 1
                    idx_result1, idx_result2 = 0, 0
                    for i in range(0, len(final_frame_lines)):
                        if final_frame_lines[i]['x_when_y_650'] == result1['x_when_y_650']:
                            idx_result1 = i
                            continue
                        if final_frame_lines[i]['x_when_y_650'] == result2['x_when_y_650']:
                            idx_result2 = i
                            continue
                    if result1['overall_score'] >= result2['overall_score']:
                        final_frame_lines.pop(idx_result2)
                        result1['inferring'] = 'inferred'

                    else:
                        final_frame_lines.pop(idx_result1)
                        result2['inferring'] = 'inferred'
    if cleaned != 0 and print_cleaning:
        print('clean {} line'.format(cleaned))


dirs = os.listdir(path)  # 得到文件夹下的所有文件名称
# dirs.sort(key=lambda x: int(x))
for dir in dirs:  # 遍历文件夹
    print('Working on dir ', dir)
    final_frame_lines = []
    scores = []
    if not os.path.isdir(path + dir):  # 判断是否是文件夹，不是文件夹才打开
        continue
    # if dir != "283":
    #     continue
    cv_para_dir = path + dir + '/cv_para/'
    dir = path + dir + '/para/'
    if not os.path.exists(cv_para_dir):
        continue
    if not os.path.exists(dir):
        continue
    files = os.listdir(dir)
    files.sort(key=lambda x: int(x[:-5]))
    for file in files:
        if os.path.isdir(file):
            continue
        fit_para_path = dir + file
        cv_para_path = fit_para_path.replace('para', 'cv_para')
        # img_file_name = file_name.replace('para/','')[:-5]+'jpg'
        # vis = cv2.imread(img_file_name)
        try:  # some video has less frames than 150
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
                para_3 = line_data['para_3']
                final_line_data['frame'] = int(file[:-5])
                final_line_data['x'] = line_data['x']
                final_line_data['y'] = line_data['y']
                final_line_data['para_3'] = para_3
                final_line_data['x_when_y_650'] = Fun_3(para_3, 650 / image_y)
                final_line_data['solid'] = True
                final_line_data['overall_score'] = line_data['score']
                final_line_data['inferring'] = 'inferred'  # from_left, from_right, inferred
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
                fit_degree1 = math.degrees(math.atan2(fy2 - fy1, fx2 - fx1 + 2 * margin)) % 180
                fit_degree2 = math.degrees(math.atan2(fy2 - fy1, fx2 - fx1 - 2 * margin)) % 180
                degree_min = min(fit_degree1, fit_degree2)
                degree_max = max(fit_degree1, fit_degree2)
                matched_cv_grid_list = []
                for cv_signle_data in cv_data:
                    x1, y1, x2, y2 = cv_signle_data['endpoiont'][0]
                    # check whether the middle point of the cv data is in the region
                    point_cv = Point((x1 + x2) / 2, (y1 + y2) / 2)
                    # print(polygon.contains(point_cv))
                    if polygon.contains(point_cv) and cv_signle_data['length'] > 20 \
                            and cv_signle_data['degree'] > degree_min and cv_signle_data['degree'] < degree_max:
                        cv_line_data = cv_signle_data['endpoiont'][0]
                        cv_grid_data = grid_cv(cv_line_data, fa, fb)
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
                if len(resampled_x) <= 7:
                    continue
                para_3 = LaneFit_3(resampled_y, resampled_x)
                error = Error_3(para_3[0], resampled_y, resampled_x)
                error = np.std(error, ddof=1)
                w = Weight(resampled_x)
                v2 = 2 * abs(para_3[0][0]) + abs(para_3[0][1])
                score = Evaluation(error / 2, v2, w)
                scores.append(score)
                if score > 0.8:
                    final_line_data['x'] = resampled_x
                    final_line_data['y'] = resampled_y
                    final_line_data['para_3'] = para_3
                    final_line_data['x_when_y_650'] = Fun_3(para_3[0], 650 / image_y)
                    final_line_data['solid'] = True
                    final_line_data['overall_score'] = score
                    final_line_data['inferring'] = 'inferred'  # from_left, from_right, inferred
                    final_frame_lines.append(final_line_data)
                    # final_frame_lines['frame'] = fit_data['frame']
                    # final_frame_lines['line_data'] = final_line_data
                    # video_all_lines.append(final_frame_lines)
        # print("final_frame_lines {}".format(final_frame_lines))

        # print("push_data {}".format(push_data))
        # with open(push_file_name,'w',encoding='utf-8') as push_file:
        #     json.dump(push_data,push_file,ensure_ascii=False)
        # push_file.close()

    # Inferring: requires at most len(files) rounds; Each round generate at most 1 along right direction and 1 along left direction.
    # In the first round, generated 'from_left' and 'frome_right' line based on infered,
    # In the rest rounds, generate 'frome_left' from 'from_left' and change the original one to 'inferred'. Do the same for 'frome_left'.
    # At the end of each round, delete conflicted lines(same frame and x_when_y_650 difference <100 pixel), and change (or keep) the remaining one to 'inferred';

    for i in range(len(files)):
        final_frame_lines_copy = copy.deepcopy(
            final_frame_lines)  # Should not't iterate the original as the original gets appended while iteration
        print('Inferring round {} with {} data'.format(i, len(final_frame_lines_copy)))
        for idx, frame_line in enumerate(final_frame_lines_copy):
            # print('In iteration {}: {} out of {}'.format(i, idx, len(final_frame_lines_copy)))
            if i == 0 and frame_line['inferring'] == 'inferred':
                if frame_line['frame'] > 1:
                    inferred_from_right = infer_line(origin_line=frame_line, infer_label='from_right')
                    final_frame_lines.append(inferred_from_right)
                if frame_line['frame'] < len(files):
                    inferred_from_left = infer_line(origin_line=frame_line, infer_label='from_left')
                    final_frame_lines.append(inferred_from_left)
            elif frame_line['inferring'] == 'from_right':
                if frame_line['frame'] > 1:
                    inferred_from_right = infer_line(origin_line=frame_line, infer_label='from_right')
                    final_frame_lines[idx]['inferring'] = 'inferred'  # Change in the original one not the copied!!!
                    # frame_line['inferring'] = 'inferred' # Change in the original one not the copied!!!
                    final_frame_lines.append(inferred_from_right)
            elif frame_line['inferring'] == 'from_left':
                if frame_line['frame'] < len(files):
                    inferred_from_left = infer_line(origin_line=frame_line, infer_label='from_left')
                    final_frame_lines[idx]['inferring'] = 'inferred'  # Change in the original one not the copied!!!
                    # frame_line['inferring'] = 'inferred' # Change in the original one not the copied!!!
                    final_frame_lines.append(inferred_from_left)
            else:
                pass
        clean_conflict_line(final_frame_lines)
    # import pdb; pdb.set_trace()
    final_frame_lines = sorted(final_frame_lines, key=lambda x: x['frame'])
    # Display lines by frame
    for i in range(1, len(files)+1):
        print('Display frame {}'.format(i+1))
        x, y = [], []
        num_lines = 0
        for data in final_frame_lines:
            if data['frame'] == i:
                num_lines += 1
                if num_lines == 3:
                    print(i)
                x = x + data['x']
                y = y + data['y']
                # show(x,y)
            if num_lines > 2:
                show(x, y, frame=data['frame'])
                # time.sleep(3)
                # plt.close()

    print(dir)
