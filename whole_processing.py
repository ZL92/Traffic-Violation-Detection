import pixellib  # pixellib requires tensorflow==2.0.0
from pixellib.instance import instance_segmentation
import cv2
import numpy as np
import glob
import os
import copy

instance_seg = instance_segmentation()
instance_seg.load_model("mask_rcnn_coco.h5")
path = "./challenge_testing_data/testing_data/video1/img/"  # 文件夹目录
folders = os.listdir(path)


def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)


def merge_car_masks(segmask):
    idx = (segmask['class_ids'] == 1) | (segmask['class_ids'] == 2) | (segmask['class_ids'] == 3) | (
            segmask['class_ids'] == 4) | (segmask['class_ids'] == 6) | (
                  segmask['class_ids'] == 8) | (segmask['class_ids'] == 99)
    mask_shape = segmask['masks'][:, :, 0].shape
    mask = np.ones(mask_shape, dtype=bool)  # Initial with True
    if len(idx) != 0:
        for i in range(len(idx)):
            if idx[i]:
                mask = mask & (~segmask['masks'][:, :, i])
        return mask.astype('uint8') * 255
    else:
        return mask


def extract_file_name(x):
    return int(x.split('/')[-1][:-4])


def remove_irrelated_results(segmasks):
    '''
    remove detection results that are not vehicles
    '''
    for segmask in segmasks:
        if len(segmask['scores']) == 1:  # This is the filled mask for batch processing. Skip this one
            continue
        for i in range(len(segmask['class_ids'])):
            for idx, id in enumerate(segmask['class_ids']):
                if id != 1 and id != 2 and id != 3 and id != 4 and id != 6 and id != 8:
                    segmask['class_ids'] = np.delete(segmask['class_ids'], idx, 0)
                    segmask['masks'] = np.delete(segmask['masks'], idx, 2)
                    segmask['rois'] = np.delete(segmask['rois'], idx, 0)
                    segmask['scores'] = np.delete(segmask['scores'], idx, 0)
                    break
                else:
                    continue


def fill_missed_mask(segmasks):
    '''In a series of the same masks (coordinates difference less than (10 pixel* num_frame difference)),
    if the mask disappear and re-appears after less than 5 frames, the mask is considered as missed.
    Note that everything in segmarks is ndarray, everything in item_attributes should be ndarray as well'''
    items = []  # List of item(vehicles) in the video
    for idx1, segmask in enumerate(segmasks):
        # print('idx1:', idx1)
        if len(segmask['scores']) == 1:  # This is the filled mask for batch processing. Skip this one
            continue
        for idx2, _ in enumerate(segmask['class_ids']):
            # print('idx2:', idx2)
            item_attributes = {
                'frame': [],
                'mask': [],
                'roi': []
            }
            Processed = False
            if idx1 == 0:  # TODO: There will be a bug if the first frame was filled for batch processing with one irrelevant result. But it doesn't exits in the dataset.
                # Add all masks in the first frame into items
                item_attributes['frame'] = idx1
                item_attributes['mask'] = segmask['masks'][idx2]
                item_attributes['roi'] = segmask['rois'][idx2]
                items.append(item_attributes)
            else:
                item_attributes = {
                    'frame': [],
                    'mask': [],
                    'roi': []
                }
                items_copy = copy.deepcopy(items)
                # update item if coordinates are closed enough; Add item if new item detected
                for idx3, item in enumerate(items_copy):
                    # Check if they are the same object
                    item_center = [(item['roi'][0] + item['roi'][1]) / 2,
                                   (item['roi'][2] + item['roi'][3]) / 2]  # [left top (y, x), right_bottom (y, x)]
                    this_center = [(segmask['rois'][idx2][0] + segmask['rois'][idx2][1]) / 2,
                                   (segmask['rois'][idx2][2] + segmask['rois'][idx2][3]) / 2]
                    if abs(item_center[0] - this_center[0]) < 10 and abs(item_center[1] - this_center[1]) < 10 and abs(
                            item['frame'] - idx1) == 1:
                        # same mask, update it
                        item_attributes['frame'] = idx1
                        item_attributes['mask'] = segmask['masks'][idx2]
                        item_attributes['roi'] = segmask['rois'][idx2]
                        items[idx3] = item_attributes
                        Processed = True
                        # print('update mask')
                    elif 1 < abs(item['frame'] - idx1) < 5 and abs(item_center[0] - this_center[0]) < 10 * abs(
                            item['frame'] - idx1) and abs(item_center[1] - this_center[1]) < 10 * abs(
                        item['frame'] - idx1):
                        # print('missed masks detected')
                        # Missing masks detected
                        frame_diff = abs(item['frame'] - idx1)
                        y_diff_per_frame = int((item_center[0] - this_center[0]) / frame_diff)
                        x_diff_per_frame = int((item_center[1] - this_center[1]) / frame_diff)
                        # Fill missed masks
                        for i in range(1, (frame_diff + 1)):
                            # TODO: Check if adding item influence iteration??
                            # Missed mask has score 1, class_ids 99, mask and roi
                            segmasks[item['frame'] + i]['scores'] = np.append(segmasks[item['frame'] + i]['scores'],
                                                                              [1], axis=0)
                            segmasks[item['frame'] + i]['class_ids'] = np.append(
                                segmasks[item['frame'] + i]['class_ids'], [99], axis=0)
                            new_roi = [item['roi'][0] + y_diff_per_frame * i,
                                       item['roi'][1] + x_diff_per_frame * i,
                                       item['roi'][2] + y_diff_per_frame * i,
                                       item['roi'][3] + x_diff_per_frame * i,
                                       ]
                            segmasks[item['frame'] + i]['rois'] = np.append(segmasks[item['frame'] + i]['rois'],
                                                                           [new_roi], axis=0)
                            maskkk = np.ones((720, 1280), dtype=bool)
                            maskkk = ~maskkk
                            maskkk[new_roi[0]:new_roi[2], new_roi[1]:new_roi[3]] = True
                            segmasks[item['frame'] + i]['masks'] = np.append(segmasks[item['frame'] + i]['masks'],
                                                                             maskkk.reshape(720, 1280, 1), axis=2)
                        Processed = True
                    else:
                        # print('else')
                        pass
                if not Processed:
                    # print('new item detected')
                    # New item detected
                    item_attributes['frame'] = idx1
                    item_attributes['mask'] = segmask['masks'][idx2]
                    item_attributes['roi'] = segmask['rois'][idx2]
                    items.append(item_attributes)
    return segmasks

def save_segmasks(segmasks, mask_folder_path):
    for i in range(len(segmasks)):
        mask = merge_car_masks(segmasks[i])
        kernel = np.ones((10, 10), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_ERODE, kernel)
        cv2.imwrite(mask_folder_path + 'corrected_{}.jpg'.format(i+1), mask)
        # print('{} saved'.format(i))

for folder in folders:
    print('In whole processing.py: path_img is ', folder)
    img_folder_path = path + folder + '/'
    paths_img = glob.glob(img_folder_path + '*.jpg')
    paths_img.sort(key=extract_file_name)
    segmasks = []
    mask_folder_path = img_folder_path + 'mask/'

    mkdir(mask_folder_path)

    for path_img in paths_img:
        segmask, output = instance_seg.segmentImage(path_img, show_bboxes=False)
        # Note: segmask items are sorted by scores!!! ROI is represented in the form of (left_top_y, left_top_x, right_bottom_y, right_bottom_x)
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

    # fill missed masks
    remove_irrelated_results(segmasks)
    segmasks = fill_missed_mask(segmasks)
    save_segmasks(segmasks, mask_folder_path)