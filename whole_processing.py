import pixellib  # pixellib requires tensorflow==2.0.0
from pixellib.instance import instance_segmentation
import cv2
import numpy as np
import glob
import os

instance_seg = instance_segmentation()
instance_seg.load_model("mask_rcnn_coco.h5")
path = "./challenge_testing_data/testing_data/video/img/"  # 文件夹目录
folders = os.listdir(path)


def merge_car_masks(segmask):
    idx = (segmask['class_ids'] == 1) | (segmask['class_ids'] == 2) | (segmask['class_ids'] == 3) | (
            segmask['class_ids'] == 4) | (segmask['class_ids'] == 6) | (
                  segmask['class_ids'] == 8)
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


for folder in folders:
    img_folder_path = "./challenge_testing_data/testing_data/video/img/" + folder + '/'
    paths_img = glob.glob(img_folder_path + '*.jpg')
    paths_img.sort(key=extract_file_name)
    segmasks = []
    mask_folder_path = img_folder_path + 'mask/'
    if not os.path.isdir(mask_folder_path):
        os.mkdir(mask_folder_path)

        for path_img in paths_img:
            print('path_img is ', path_img)
            segmask, output = instance_seg.segmentImage(path_img,
                                                        show_bboxes=False)  # segmask items are sorted by scores!!! ROI is represented in the form of (left_top_y, left_top_x, right_bottom_y, right_bottom_x)
            if len(segmask['scores']) == 0:
                print('The empty detection gets filled: {}'.format(paths_img))
                segmask['class_ids'] = np.append(segmask['class_ids'], np.asarray((15)))
                segmask['scores'] = np.append(segmask['scores'], np.asarray((0.9)))
                segmask['rois'] = np.append(segmask['rois'], np.asarray([(529, 359, 531, 361)])).reshape(1, 4)
                fake_mask = np.ones((1280, 720), dtype=bool)
                fake_mask[529:531, 359:361] = False
                segmask['masks'] = np.append(segmask['masks'], fake_mask).reshape(720, 1280, 1)
                segmask['masks'] = segmask['masks'] < 0.5

            segmasks.append(segmask)

        # fill missed masks

        for i in range(1, len(segmasks)):
            previous_roi_center = [((a + b) / 2, (c + d) / 2) for a, b, c, d in segmasks[i - 1]['rois']]
            current_roi_center = [((a + b) / 2, (c + d) / 2) for a, b, c, d in segmasks[i]['rois']]

            for idx in range(len(previous_roi_center)):
                coords = previous_roi_center[idx]
                if 10 < coords[0] < 710 and 10 < coords[1] < 1270:
                    mask_missed = False
                    for y_x in current_roi_center:
                        # criteria: not close to borders; the distance difference to the previous frame is less than 10 pixel (one example is that a mask varies in three frames as (442.5, 626.0) to (441.0, 627.5) to (439.5, 630.0) )
                        if (y_x[0] - 10 < coords[0] < y_x[0] + 10) and (y_x[1] - 10 < coords[1] < y_x[1] + 10):
                            # print('found {}'.format(coords))
                            detected = True
                            break
                        else:
                            detected = False
                    if not detected:
                        mask_missed = True
                if mask_missed:
                    print('mask {} in frame{} is missed in frame{}'.format(idx, i - 1, i))
                    # print('shapes of class_ids, rois, scores, masks are {} {} {} {}'.format(segmasks[i]['class_ids'].shape,
                    #                                                                         segmasks[i]['rois'].shape,
                    #                                                                         segmasks[i]['scores'].shape,
                    #                                                                         segmasks[i]['masks'].shape))
                    segmasks[i]['class_ids'] = np.append(segmasks[i]['class_ids'],
                                                         np.asarray([segmasks[i - 1]['class_ids'][idx]]), axis=0)
                    segmasks[i]['rois'] = np.append(segmasks[i]['rois'], np.asarray([segmasks[i - 1]['rois'][idx]]),
                                                    axis=0)
                    segmasks[i]['scores'] = np.append(segmasks[i]['scores'],
                                                      np.asarray([segmasks[i - 1]['scores'][idx]]),
                                                      axis=0)
                    segmasks[i]['masks'] = np.append(segmasks[i]['masks'],
                                                     np.asarray(segmasks[i - 1]['masks'][:, :, idx]).reshape(720, 1280,
                                                                                                             1),
                                                     axis=2)
                    # print('After adding, shapes of class_ids, rois, scores, masks are {} {} {} {}'.format(
                    #     segmasks[i]['class_ids'].shape,
                    #     segmasks[i]['rois'].shape,
                    #     segmasks[i]['scores'].shape,
                    #     segmasks[i]['masks'].shape))
        for i in range(len(segmasks)):
            mask = merge_car_masks(segmasks[i])
            kernel = np.ones((10, 10), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_ERODE, kernel)
            cv2.imwrite(mask_folder_path + 'corrected_{}.jpg'.format(i), mask)
            print('{} saved'.format(i))
