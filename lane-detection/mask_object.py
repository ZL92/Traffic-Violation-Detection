import pixellib
from pixellib.instance import instance_segmentation
import cv2
import numpy as np
import glob


def merge_car_masks(segmask):
    idx = (segmask['class_ids'] == 2) | (segmask['class_ids'] == 3) | (segmask['class_ids'] == 4) | (segmask['class_ids'] == 1) | (segmask['class_ids'] == 7) | (segmask['class_ids'] == 8) # 2 represents bicycle, 3 represents car, 4 represents motorcycle,
    mask_shape = segmask['masks'][:, :, 0].shape
    mask = np.ones(mask_shape, dtype=bool)  # Initial with True
    for i in range(len(idx)):
        if idx[i]:
            mask = mask & (~segmask['masks'][:, :, i])
    return mask.astype('uint8') * 255

instance_seg = instance_segmentation()
instance_seg.load_model("/home/gym/mask_rcnn_coco.h5")
img_folder_path = "/home/gym/video/img/283/"
paths_img = glob.glob(img_folder_path+'/*.jpg')
for path_img in paths_img:
    print('path_img is ', path_img)
    segmask, output = instance_seg.segmentImage(path_img, show_bboxes=False)
    # cv2.imwrite("instance_seg_{}".format(path_img[-5:]), output)
    # print(output.shape)
    mask = merge_car_masks(segmask)
    #Erode mask to fill holes in mask
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_ERODE, kernel)
    print(mask)
    img_name = 'home/gym/video/img/283/masks_folder/{}.png'.format(path_img.split('/')[-4])
    print(img_name)
    cv2.imwrite(img_name, mask)
