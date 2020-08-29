import torch, os, cv2
from model.model import parsingNet
from utils.common import merge_config
from utils.dist_utils import dist_print
import torch
import scipy.special, tqdm
import numpy as np
import json
import torchvision.transforms as transforms
from data.dataset import LaneTestDataset
from data.constant import culane_row_anchor, tusimple_row_anchor
def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)   
if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True

    args, cfg = merge_config()

    dist_print('start testing...')
    assert cfg.backbone in ['18','34','50','101','152','50next','101next','50wide','101wide']

    if cfg.dataset == 'CULane':
        cls_num_per_lane = 18
    elif cfg.dataset == 'Tusimple':
        cls_num_per_lane = 56
    else:
        raise NotImplementedError

    net = parsingNet(pretrained = False, backbone=cfg.backbone,cls_dim = (cfg.griding_num+1,cls_num_per_lane,4),
                    use_aux=False) # we dont need auxiliary segmentation in testing

    state_dict = torch.load(cfg.test_model, map_location='cpu')['model']
    compatible_state_dict = {}
    for k, v in state_dict.items():
        if 'module.' in k:
            compatible_state_dict[k[7:]] = v
        else:
            compatible_state_dict[k] = v

    net.load_state_dict(compatible_state_dict, strict=False)
    net.eval()

    img_transforms = transforms.Compose([
        transforms.Resize((288, 800)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    if cfg.dataset == 'CULane':
        splits = ['test0_normal.txt', 'test1_crowd.txt', 'test2_hlight.txt', 'test3_shadow.txt', 'test4_noline.txt', 'test5_arrow.txt', 'test6_curve.txt', 'test7_cross.txt', 'test8_night.txt']
        datasets = [LaneTestDataset(cfg.data_root,os.path.join(cfg.data_root, 'list/test_split/'+split),img_transform = img_transforms) for split in splits]
        img_w, img_h = 1640, 590
        row_anchor = culane_row_anchor
    elif cfg.dataset == 'Tusimple':
        splits = ['test.txt']
        datasets = [LaneTestDataset(cfg.data_root,os.path.join(cfg.data_root, split),img_transform = img_transforms) for split in splits]
        img_w, img_h = 1280, 720
        row_anchor = tusimple_row_anchor
    else:
        raise NotImplementedError
    for split, dataset in zip(splits, datasets):
        loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle = False, num_workers=1)
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        print(cfg.data_root+split[:-3]+'avi')
        vout = cv2.VideoWriter(cfg.data_root+split[:-3]+'avi', fourcc , 30.0, (img_w, img_h))
        vout_empty = cv2.VideoWriter(cfg.data_root+'empty_'+split[:-3]+'avi', fourcc , 30.0, (img_w, img_h))
        mkdir(cfg.data_root+'line/')
        for i, data in enumerate(tqdm.tqdm(loader)):
            imgs, names = data
            f = open(cfg.data_root+'line/'+names[0][:-3]+'json','w+')
            imgs = imgs
            #print("imgs size {}".format(imgs.size()))
            with torch.no_grad():
                out = net(imgs)

            #print("out size {}".format(out.size()))
            col_sample = np.linspace(0, 800 - 1, cfg.griding_num)
            col_sample_w = col_sample[1] - col_sample[0]


            out_j = out[0].data.cpu().numpy()
            #print("out_j\n {}".format(out_j))
            out_j = out_j[:, ::-1, :]
            
            #print("out_j type {}".format(type(out_j)))
            prob = scipy.special.softmax(out_j[:-1, :, :], axis=0)
            #print("prob shape {}".format(prob.shape))
            idx = np.arange(cfg.griding_num) + 1
            #print("idx {}".format(idx))
            idx = idx.reshape(-1, 1, 1)
            #print("reshape idx {}".format(idx))
            loc = np.sum(prob * idx, axis=0)
            #print("loc {}".format(loc))
            out_j = np.argmax(out_j, axis=0)
            #print("out_j type {}".format(type(out_j)))
            loc[out_j == cfg.griding_num] = 0
            out_j = loc
            prob[prob<0.01] = 0
            # import pdb; pdb.set_trace()
            vis = cv2.imread(os.path.join(cfg.data_root,names[0]))
            emptyImage = np.zeros(vis.shape,np.uint8)
            #print(names[0])
            result = []
            for i in range(out_j.shape[1]):
                if i == 0:
                    color = (255,0,0)
                elif i == 1:
                    color = (0,255,0)
                elif i == 2:
                    color = (0,0,255)
                else:
                    color = (100,100,100)
                if np.sum(out_j[:, i] != 0) > 2:
                    for k in range(out_j.shape[0]):
                        temp_dict = {}
                        if out_j[k, i] > 0:
                            #print("prob k,i {}".format(prob[:,k,i]))
                            ppp = (int(out_j[k, i] * col_sample_w * img_w / 800) - 1, int(img_h * (row_anchor[cls_num_per_lane-1-k]/288)) - 1 )
                            temp_dict['lane_id'] = i
                            temp_dict['pos'] = ppp
                            temp_dict['prob'] = prob[:,k,i].tolist()
                            result.append(temp_dict)
                            cv2.circle(vis,ppp,5,(0,255,0),-1)
                            cv2.circle(emptyImage,ppp,5,color,-1)
            vout.write(vis)
            vout_empty.write(emptyImage)
            json.dump(result,f)
            f.close()
        vout.release()
