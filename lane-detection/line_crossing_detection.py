'''
Output example:
[{"start_time": 1.73, "xs": 100, "ys": 323, "ws": 321, "hs": 323, "end_time": 2.33, "xe": 224, "ye": 222, "we": 222, "he": 229, "line_style": "solid"},
{"start_time": 1.73, "xs": 818, "ys": 111, "ws": 111, "hs": 111, "end_time": 4.87, "xe": 1111, "ye": 111, "we": 111, "he": 111, "line_style": "solid"}]

(xs, ys) is the center of the box, ws and hs are weight and height of the box.

'''

from pixellib.instance import instance_segmentation
import cv2
import numpy as np
import glob
import os
import json
import copy

instance_seg = instance_segmentation()
instance_seg.load_model("mask_rcnn_coco.h5")
path = "./video1/img/"  # 文件夹目录
folders = os.listdir(path)


#


def extract_file_name(x):
    return int(x.split('/')[-1][:-4])

def Fun_3(p, y):
    a1, a2, a3, a4 = p
    return a1 * y ** 3 + a2 * y ** 2 + a3 * y + a4

for folder in folders:
    print('In line_crossing_detection.py: ', folder)
    img_folder_path = path + folder + '/'
    paths_img = glob.glob(img_folder_path + '*.jpg')
    paths_img.sort(key=extract_file_name)
    segmasks=[]

    # Vehicle detection
    for path_img in paths_img:
        # print('path_img is ', path_img)
        segmask, output = instance_seg.segmentImage(path_img,show_bboxes=False)
        # segmask items are sorted by scores!!! ROI is represented in the form of (left_top_y, left_top_x, right_bottom_y, right_bottom_x)
        if len(segmask[
                   'scores']) == 0:  # For batch processing, when there is a non-value segmask, fill an irrelevant result
            # print('The empty detection gets filled: {}'.format(paths_img))
            segmask['class_ids'] = np.append(segmask['class_ids'], np.asarray((15)))
            segmask['scores'] = np.append(segmask['scores'], np.asarray((0.9)))
            segmask['rois'] = np.append(segmask['rois'], np.asarray([(529, 359, 531, 361)])).reshape(1, 4)
            fake_mask = np.ones((1280, 720), dtype=bool)
            fake_mask[529:531, 359:361] = False
            segmask['masks'] = np.append(segmask['masks'], fake_mask).reshape(720, 1280, 1)
            segmask['masks'] = segmask['masks'] < 0.5

        segmasks.append(segmask)


    # Load line data
    line_data_path = img_folder_path + 'corrected_lines_data_final/'
    paths_json = os.listdir(line_data_path)
    paths_json.sort(key=lambda x:int(x[:-5]))
    json_data_list = []
    for path_json in paths_json:
        f = open(line_data_path+path_json, 'r')
        line_data = json.load(f)
        json_data_list.append(line_data)

    # Calculate roi of each detected item (vehicle)
    items = []  # List of item(vehicles) in the video
    for idx1, segmask in enumerate(segmasks):
        # print('idx1:', idx1)
        if len(segmask['scores']) == 1:  # This is the filled mask for batch processing. Skip this one
            continue
        for idx2, _ in enumerate(segmask['class_ids']):
            # print('idx2:', idx2)
            item_attributes = {
                'frame': [],
                'roi': []
            }
            Processed = False
            if idx1 == 0:
                item_attributes['frame'].append(idx1 + 1)
                item_attributes['roi'].append(segmask['rois'][idx2].tolist())
                items.append(item_attributes)
            else:
                for idx3, item in enumerate(items):
                    item_center = [(item['roi'][-1][0] + item['roi'][-1][1]) / 2,
                                   (item['roi'][-1][2] + item['roi'][-1][3]) / 2]  # [left top (y, x), right_bottom (y, x)]
                    this_center = [(segmask['rois'][idx2][0] + segmask['rois'][idx2][1]) / 2,
                                   (segmask['rois'][idx2][2] + segmask['rois'][idx2][3]) / 2]
                    if abs(item_center[0] - this_center[0]) < 10 and abs(item_center[1] - this_center[1]) < 10 and abs(
                            item['frame'][-1] - idx1) == 1:
                        # same mask, update it
                        item['frame'].append(idx1 + 1)
                        item['roi'].append(segmask['rois'][idx2].tolist())
                        Processed = True
                        # print('update mask')
                    elif 1 < abs(item['frame'][-1] - idx1) < 5 and abs(item_center[0] - this_center[0]) < 10 * abs(
                            item['frame'][-1] - idx1) and abs(item_center[1] - this_center[1]) < 10 * abs(
                        item['frame'][-1] - idx1):
                        # print('missed masks detected')
                        # Missing masks detected
                        frame_diff = abs(item['frame'][-1] - idx1)
                        y_diff_per_frame = int((item_center[0] - this_center[0]) / frame_diff)
                        x_diff_per_frame = int((item_center[1] - this_center[1]) / frame_diff)
                        # Fill missed masks
                        for i in range(1, (frame_diff + 1)):
                            start_frame_id = item['frame'][-1]
                            item['frame'].append(start_frame_id+i)
                            new_roi = [item['roi'][-1][0] + y_diff_per_frame * i,
                                       item['roi'][-1][1] + x_diff_per_frame * i,
                                       item['roi'][-1][2] + y_diff_per_frame * i,
                                       item['roi'][-1][3] + x_diff_per_frame * i,
                                       ]
                            item['roi'].append(new_roi)
                        Processed = True
                    else:
                        pass
                if not Processed:
                    # print('new item detected')
                    # New item detected
                    item_attributes['frame'].append(idx1 + 1)
                    item_attributes['roi'].append(segmask['rois'][idx2].tolist())
                    items.append(item_attributes)
    if len(items)==0:
        with open('./test_' + folder + '.json', 'w', encoding='utf-8') as push_file:
            json.dump(items, push_file, ensure_ascii=False)
        break
    # Detect crossing
    results=[]
    for item in items:
        Start_crossing = False
        result_dict = {
            "start_frame": [],
            "start_roi": [],
            "end_frame": [],
            'end_roi': [],
            'solid': []
        }
        for frame_id, roi in zip(item['frame'], item['roi']):
            check_line_left = [roi[1] + (roi[3] - roi[1])/5, roi[0]+(roi[2] - roi[0])*3/4]
            check_line_right = [roi[1] + (roi[3] - roi[1])*4/5, roi[0]+(roi[2] - roi[0])*3/4]
            check_line = [check_line_left, check_line_right] #[[x1,y1], [x2,y2]]

            try:
                this_lines_data = json_data_list[frame_id-1] #json data is in range 0~1
            except:# TODO: Here is a bug in frame_id. It exceeds 149 sometimes
                # print('except frame id: ', frame_id)
                # break
                this_lines_data=json_data_list[-1]
            for this_line_data in this_lines_data:
                para_3 = this_line_data['para_3']
                x_when_same_y = Fun_3(para_3, (roi[0]+(roi[2] - roi[0])*3/4)/720) * 1280
                if (not Start_crossing) and check_line_left[0]< x_when_same_y < check_line_right[0]:
                    Start_crossing = True
                    result_dict['start_frame'] = frame_id
                    result_dict['start_roi'] = roi
                    result_dict['solid'] = this_line_data['solid']
                elif Start_crossing and (not check_line_left[0]< x_when_same_y<check_line_right[0]):
                    result_dict['end_roi'] = roi
                    result_dict['end_frame'] = frame_id
                    results.append(result_dict)
                    result_dict = {
                        "start_frame": [],
                        "start_roi": [],
                        "end_frame": [],
                        'end_roi': [],
                        'solid': []
                    }
                    Start_crossing = False
                else:
                    continue
    # Change results format and save json
    final_results_json = []
    for single_result in results:
        single_result_json = {
            'start_time':[],
            'xs':[],
            'ys':[],
            'ws':[],
            'hs':[],
            'end_time':[],
            'xe':[],
            'ye':[],
            'we':[],
            'he':[],
            'line_style':[]
        }
        fps=30

        single_result_json['start_time'] = round(single_result['start_frame']/fps,2)
        single_result_json['xs'] = (single_result['start_roi'][1] + single_result['start_roi'][3])/2
        single_result_json['ys'] = (single_result['start_roi'][0] + single_result['start_roi'][2]) / 2
        single_result_json['ws'] = single_result['start_roi'][3] - single_result['start_roi'][1]
        single_result_json['hs'] = single_result['start_roi'][2] - single_result['start_roi'][0]

        single_result_json['end_time'] = round(single_result['end_frame']/fps, 2)
        single_result_json['xe'] = (single_result['end_roi'][1] + single_result['end_roi'][3])/2
        single_result_json['ye'] = (single_result['end_roi'][0] + single_result['end_roi'][2]) / 2
        single_result_json['we'] = single_result['end_roi'][3] - single_result['end_roi'][1]
        single_result_json['he'] = single_result['end_roi'][2] - single_result['end_roi'][0]

        if single_result['solid']:
            single_result_json['line_style'] = 'solid'
        else:
            single_result_json['line_style'] = 'dash'

        final_results_json.append(single_result_json)
    # Delete duplicated result
    if len(final_results_json) == 0:
        pass
    else:
        Deleting = True
        idx=0
        while Deleting:
            to_be_removed_items=[]
            for a, b in enumerate(final_results_json):
                if a !=idx and abs(final_results_json[idx]['xs'] - b['xs'])<20 and abs(final_results_json[idx]['ys'] - b['ys'])<20 and abs(final_results_json[idx]['start_time'] - b['start_time'])<0.3:
                    to_be_removed_items.append(b)
                else:
                    pass
            for to_be_removed_item in to_be_removed_items:
                final_results_json.remove(to_be_removed_item)
            idx +=1
            if idx == len(final_results_json):
                Deleting = False


    with open('./' + folder + '.json','w', encoding='utf-8') as push_file:
            json.dump(final_results_json, push_file, ensure_ascii=False)
