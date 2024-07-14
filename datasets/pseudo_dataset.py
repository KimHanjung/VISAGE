import os
import json
import sys
import random
import itertools
import cv2

import numpy as np
from pycocotools import mask as maskUtils
from glob import glob
from copy import deepcopy

import random
from pycocotools.coco import COCO
from detectron2.data.datasets.builtin_meta import _get_coco_instances_meta
from utils import get_valid_pairs

COCO_TO_YTVIS_2019 = {
    1: 1, 2: 21, 3: 6, 4: 21, 5: 28, 7: 17, 8: 29, 9: 34, 17: 14, 18: 8, 19: 18, 21: 15, 22: 32, 23: 20, 24: 30, 25: 22,
    35: 33, 36: 33, 41: 5, 42: 27, 43: 40
}

total_classes = [1, 1, 2, 3, 4, 5, 7, 8, 9, 17, 18, 19, 21, 22, 23, 24, 25, 35, 36, 41, 42, 43] # ytvis distribution
print("total class number: ", len(set(total_classes)))
for c in set(total_classes):
    assert c in COCO_TO_YTVIS_2019.keys(), c
    print("Class probability for class {}: {}%".format(c, int(100*total_classes.count(c)/len(total_classes))))
    
coco_meta = _get_coco_instances_meta()

coco = COCO('/workspace/datasets/coco/annotations/instances_train2017.json')
coco_root = '/workspace/datasets/coco/train2017'

def polygonFromMask(maskedArr): 

        contours, _ = cv2.findContours(maskedArr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        segmentation = []
        for contour in contours:
            # Valid polygons have >= 6 coordinates (3 points)
            if contour.size >= 6:
                segmentation.append(contour.flatten().tolist())
        RLEs = maskUtils.frPyObjects(segmentation, maskedArr.shape[0], maskedArr.shape[1])
        RLE = maskUtils.merge(RLEs)
        area = maskUtils.area(RLE)
        [x, y, w, h] = cv2.boundingRect(maskedArr)

        return segmentation[0], [x, y, w, h], area

# for bezier curve calculation
def bezier_curve(t, p0, p1, p2):
    return (1-t)**2 * p0 + 2 * (1-t) * t * p1 + t**2 * p2

# initialize random points for bezier curve
def generate_random_points(width_diff, height_diff, w, h):

    p0 = np.random.rand(2)
    p1 = np.random.rand(2)
    p2 = np.random.rand(2)

    while height_diff - p0[0] * h < 0 or width_diff - p0[1] * w < 0:
        p0 = np.random.rand(2)
    while height_diff - p2[0] * h < 0 or width_diff - p2[1] * w < 0:
        p2 = np.random.rand(2)
        
    return p0, p1, p2

def make_pseudo_video(root_path, vid_name, bg_imgs, vid_id, start_anno_id, tgt, tgt_cat, change_num, margin_pixel):
    h, w = bg_imgs[0].shape[:2]
    length = len(bg_imgs)

    video = {
        'width': w - 2*margin_pixel,
        'height': h - 2*margin_pixel,
        'length': 36,
        'license': '',
        'date_captured': '',
        'flickr_url': '',
        'coco_url': '',
        'id': vid_id,
        'file_names': [os.path.join(vid_name, f'{idx:04d}.jpg') for idx in range(length)],
    }
    yt_annotations = []
    video_mask = np.zeros((length, h, w, 1), dtype=np.uint8)

    tracks = []
    starts = []
    
    maxy, maxx = 0, 0
    for i, ((img_id, ann_id), cat) in enumerate(zip(tgt, tgt_cat)):

        tgt_anno = coco.loadAnns(ann_id)[0]
        tgt_img = cv2.imread(os.path.join(coco_root, coco.loadImgs(img_id)[0]['file_name']))
        tgt_mask = coco.annToMask(tgt_anno)
        tgt_bbox = np.array(
        [np.where(tgt_mask)[1].min(), np.where(tgt_mask)[0].min(), np.where(tgt_mask)[1].max()+1, np.where(tgt_mask)[0].max()+1]
        )
    
        cropped_img = tgt_img[tgt_bbox[1]:tgt_bbox[3], tgt_bbox[0]:tgt_bbox[2]]
        maxy = max(maxy, cropped_img.shape[0])
        maxx = max(maxx, cropped_img.shape[1])
    
    height_diff = bg_imgs[0].shape[0] - maxy
    width_diff = bg_imgs[0].shape[1] - maxx
    
    for i, ((img_id, ann_id), cat) in enumerate(zip(tgt, tgt_cat)):
        t_values = np.linspace(0, 1, length+1) 
        delta_list = []
        cum_delta_list = []

        p0, p1, p2 = generate_random_points(width_diff, height_diff, w, h) 
        start_y, start_x = (int(p0[0] * h), int(p0[1] * w))
        starts.append([(start_y, start_x) for _ in range(length)])
        last = None
        
        for frame in range(length+1):
            coordinates = bezier_curve(t_values[frame], p0, p1, p2) 
            coordinates_pixel = (int(coordinates[0] * h), int(coordinates[1] * w))  
            
            if last is None:
                last = coordinates_pixel
                cur = [0,0]
            else:
                delta_list.append((coordinates_pixel[0] - last[0], coordinates_pixel[1] - last[1]))
                cur[0] += coordinates_pixel[0] - last[0]
                cur[1] += coordinates_pixel[1] - last[1]
                cum_delta_list.append((cur[0], cur[1]))
                last = coordinates_pixel
                
        tracks.append(cum_delta_list)

    #######
    # track shuffle
    
    idxs = random.sample(range(1, length-1), change_num)
    print("shuffled at frame :", sorted(idxs))
    for idx in sorted(idxs):
        newtracks = deepcopy(tracks)
        newstarts = deepcopy(starts)
    
        order = list(range(0, len(tracks)))
        random.shuffle(order)
        
        for i, o in enumerate(order):
            newtracks[o][idx:] = tracks[i][idx:]
            newstarts[o][idx:] = starts[i][idx:]
            
        tracks = newtracks
        starts = newstarts

    #######
    
    for i, ((img_id, ann_id), cat) in enumerate(zip(tgt, tgt_cat)):
        tgt_anno = coco.loadAnns(ann_id)[0]
        tgt_img = cv2.imread(os.path.join(coco_root, coco.loadImgs(img_id)[0]['file_name']))
        
        bg_imgs, object_anno, video_mask = get_pseudo_video_by_roll(length, bg_imgs, tgt_img, tgt_anno, video_mask, tracks[i], starts[i], height_diff, width_diff, margin_pixel)
        yt_annotations.append({
            'video_id': vid_id,
            'category_id': COCO_TO_YTVIS_2019[cat],
            'iscrowd': False,
            'id': start_anno_id,
            'height': h - 2*margin_pixel,
            'width': w - 2*margin_pixel,
            'length': 36,
        })
        yt_annotations[-1].update(object_anno)
        start_anno_id += 1

    if not os.path.exists(os.path.join(root_path, vid_name)):
        os.makedirs(os.path.join(root_path, vid_name))

    for idx, img in enumerate(bg_imgs):
        img = img[margin_pixel:-margin_pixel, margin_pixel:-margin_pixel]
        cv2.imwrite(os.path.join(root_path, video['file_names'][idx]), img)

    return video, yt_annotations

def get_pseudo_video_by_roll(length, bg_imgs, src_img, tgt_anno, video_mask, track, start, height_diff, width_diff, margin_pixel):
    h, w = bg_imgs[0].shape[:2]
    object_annotation = {
        'segmentations': [],
        'bboxes': [],
        'areas': [],
    }

    pseudo_video = []

    tgt_mask = coco.annToMask(tgt_anno)
    tgt_bbox = np.array(
        [np.where(tgt_mask)[1].min(), np.where(tgt_mask)[0].min(), np.where(tgt_mask)[1].max()+1, np.where(tgt_mask)[0].max()+1]
        )
    
    cropped_img = src_img[tgt_bbox[1]:tgt_bbox[3], tgt_bbox[0]:tgt_bbox[2]]
    cropped_mask = tgt_mask[tgt_bbox[1]:tgt_bbox[3], tgt_bbox[0]:tgt_bbox[2]]

    height_diff = bg_imgs[0].shape[0] - cropped_img.shape[0]
    width_diff = bg_imgs[0].shape[1] - cropped_img.shape[1]
    assert height_diff >= 0 and width_diff >= 0, 'background image should be larger than target image'

    
    height_max_step = h
    width_max_step = w

    pad_y = (h, h)
    pad_x = (w, w)
    
    for idx in range(length):
        
        padded_mask_ori = np.pad(cropped_mask, ((start[idx][0], height_diff-start[idx][0]), (start[idx][1], width_diff-start[idx][1])), 'constant', constant_values=0)[:, :, np.newaxis]
        padded_img_ori = np.pad(cropped_img, ((start[idx][0], height_diff-start[idx][0]), (start[idx][1], width_diff-start[idx][1]), (0, 0)), 'constant', constant_values=0)
    
        height_step = track[idx][0]
        width_step = track[idx][1]
        
        padded_mask = np.roll(
            np.pad(padded_mask_ori, (pad_y, pad_x, (0, 0)), 'constant', constant_values=0), (height_step, width_step), axis=(0, 1)
            )
        padded_mask = padded_mask[height_max_step:-height_max_step, width_max_step:-width_max_step, :]
        
        padded_img = np.roll(
            np.pad(padded_img_ori, (pad_y, pad_x, (0, 0)), 'constant', constant_values=0), (height_step, width_step), axis=(0, 1)
            )
        padded_img = padded_img[height_max_step:-height_max_step, width_max_step:-width_max_step, :]
        

        anno_mask = padded_mask & ~video_mask[idx] # handle occlusion case
        new_img = anno_mask * padded_img + (1 - anno_mask) * bg_imgs[idx]
        pseudo_video.append(new_img)

        video_mask[idx] = video_mask[idx] | anno_mask
        if anno_mask.sum() == 0:
            object_annotation['segmentations'].append(None)
            object_annotation['bboxes'].append(None)
            object_annotation['areas'].append(None)
        else:
            rle_anno = maskUtils.encode(np.array(anno_mask[margin_pixel:-margin_pixel, margin_pixel:-margin_pixel, 0], order='F')) # bitmask to rle
            rle_anno['counts'] = str(rle_anno['counts'], 'utf-8')
            object_annotation['segmentations'].append(rle_anno)
            object_annotation['bboxes'].append(maskUtils.toBbox(rle_anno))
            object_annotation['areas'].append(maskUtils.area(rle_anno))
        
    return pseudo_video, object_annotation, video_mask

# for json serialization
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            print(obj)
            return super(NpEncoder, self).default(obj)

pseudo_anno_total = {
    'info': {},
    'licenses': [],
    'videos': [],
    'annotations': [],
    'categories': [
        {"color": [220, 20, 60], "isthing": 1, "id": 1, "name": "person"},
        {"color": [0, 82, 0], "isthing": 1, "id": 2, "name": "giant_panda"},
        {"color": [119, 11, 32], "isthing": 1, "id": 3, "name": "lizard"},
        {"color": [165, 42, 42], "isthing": 1, "id": 4, "name": "parrot"},
        {"color": [134, 134, 103], "isthing": 1, "id": 5, "name": "skateboard"},
        {"color": [0, 0, 142], "isthing": 1, "id": 6, "name": "sedan"},
        {"color": [255, 109, 65], "isthing": 1, "id": 7, "name": "ape"},
        {"color": [0, 226, 252], "isthing": 1, "id": 8, "name": "dog"},
        {"color": [5, 121, 0], "isthing": 1, "id": 9, "name": "snake"},
        {"color": [0, 60, 100], "isthing": 1, "id": 10, "name": "monkey"},
        {"color": [250, 170, 30], "isthing": 1, "id": 11, "name": "hand"},
        {"color": [100, 170, 30], "isthing": 1, "id": 12, "name": "rabbit"},
        {"color": [179, 0, 194], "isthing": 1, "id": 13, "name": "duck"},
        {"color": [255, 77, 255], "isthing": 1, "id": 14, "name": "cat"},
        {"color": [120, 166, 157], "isthing": 1, "id": 15, "name": "cow"},
        {"color": [73, 77, 174], "isthing": 1, "id": 16, "name": "fish"},
        {"color": [0, 80, 100], "isthing": 1, "id": 17, "name": "train"},
        {"color": [182, 182, 255], "isthing": 1, "id": 18, "name": "horse"},
        {"color": [0, 143, 149], "isthing": 1, "id": 19, "name": "turtle"},
        {"color": [174, 57, 255], "isthing": 1, "id": 20, "name": "bear"},
        {"color": [0, 0, 230], "isthing": 1, "id": 21, "name": "motorbike"},
        {"color": [72, 0, 118], "isthing": 1, "id": 22, "name": "giraffe"},
        {"color": [255, 179, 240], "isthing": 1, "id": 23, "name": "leopard"},
        {"color": [0, 125, 92], "isthing": 1, "id": 24, "name": "fox"},
        {"color": [209, 0, 151], "isthing": 1, "id": 25, "name": "deer"},
        {"color": [188, 208, 182], "isthing": 1, "id": 26, "name": "owl"},
        {"color": [145, 148, 174], "isthing": 1, "id": 27, "name": "surfboard"},
        {"color": [106, 0, 228], "isthing": 1, "id": 28, "name": "airplane"},
        {"color": [0, 0, 70], "isthing": 1, "id": 29, "name": "truck"},
        {"color": [199, 100, 0], "isthing": 1, "id": 30, "name": "zebra"},
        {"color": [166, 196, 102], "isthing": 1, "id": 31, "name": "tiger"},
        {"color": [110, 76, 0], "isthing": 1, "id": 32, "name": "elephant"},
        {"color": [133, 129, 255], "isthing": 1, "id": 33, "name": "snowboard"},
        {"color": [0, 0, 192], "isthing": 1, "id": 34, "name": "boat"},
        {"color": [183, 130, 88], "isthing": 1, "id": 35, "name": "shark"},
        {"color": [130, 114, 135], "isthing": 1, "id": 36, "name": "mouse"},
        {"color": [107, 142, 35], "isthing": 1, "id": 37, "name": "frog"},
        {"color": [0, 228, 0], "isthing": 1, "id": 38, "name": "eagle"},
        {"color": [174, 255, 243], "isthing": 1, "id": 39, "name": "earless_seal"},
        {"color": [255, 208, 186], "isthing": 1, "id": 40, "name": "tennis_racket"}
    ],
}

pseudo_anno_track = deepcopy(pseudo_anno_total)
pseudo_anno_swap = deepcopy(pseudo_anno_total)



NUM_VIDEOS = 1000
OUTPUT_DIR = './pseudo_videos'
length = 36
start_vid = 1
start_annid = 1

# pre-selcted instances
valid_instances = get_valid_pairs()
print("number of classes : ", len(valid_instances.keys()))

# shuffle vid ids
idx = list(range(NUM_VIDEOS))
random.shuffle(idx)

# setup resolution & video type
r = range(600, 901, 100)
margin_ratio = 0.2
resolutions = [(x, y, 3) for x, y in itertools.product(r, r)]
index2vidtype = {}
video_type = [('natural', 0), ('natural', 1)] # (background_type, swap_num)

for i in range(2):
    for j in idx[i*(NUM_VIDEOS//2):(i+1)*(NUM_VIDEOS//2)]:
        index2vidtype[j] = video_type[i]


for n in range(NUM_VIDEOS):

    bg_img_list = glob('/workspace/datasets/BG-20k/train/*')
    background_image = cv2.imread(random.choice(bg_img_list))
    background_image = cv2.resize(background_image, random.choice(resolutions)[:2])

    
    # padding for margin (will be cropped later)
    margin_pixel = int(min(background_image.shape[:2]) * margin_ratio)
    background_image = cv2.copyMakeBorder(background_image, margin_pixel, margin_pixel, margin_pixel, margin_pixel, cv2.BORDER_CONSTANT, value=[0,0,0])
    
    bg_imgs = [background_image] * length
    num_instances = random.randint(2,3)
    classes = [random.choice(total_classes) for i in range(num_instances)]
    
    # sample instances
    tgt = []
    tgt_cat = []
    bbox_h, bbox_w = 0, 0
    for class_id in set(classes):
        sampled_instance = random.sample(valid_instances[class_id], classes.count(class_id))
        tgt.extend(sampled_instance)
        tgt_cat.extend([class_id] * len(sampled_instance))
    print("Video :", n)

    vids, yt_anns = make_pseudo_video(OUTPUT_DIR, str(n).zfill(4), bg_imgs, start_vid, start_annid, tgt, tgt_cat, change_num=index2vidtype[n][1], margin_pixel=margin_pixel)
    
    # update annotations (all)
    pseudo_anno_total['videos'].append(vids)
    pseudo_anno_total['annotations'].extend(yt_anns)
    
    # update annotations (swap, track)
    if index2vidtype[n][1] == 0:
        pseudo_anno_track['videos'].append(vids)
        pseudo_anno_track['annotations'].extend(yt_anns)
    elif index2vidtype[n][1] == 1:
        pseudo_anno_swap['videos'].append(vids)
        pseudo_anno_swap['annotations'].extend(yt_anns)
    else:
        raise NotImplementedError
    
    start_vid += 1
    start_annid += len(yt_anns)
    


with open(os.path.join(OUTPUT_DIR, 'annotations_all.json'), "w") as outfile:
    json.dump(pseudo_anno_total, outfile, cls=NpEncoder)
with open(os.path.join(OUTPUT_DIR, 'annotations_track.json'), "w") as outfile:
    json.dump(pseudo_anno_track, outfile, cls=NpEncoder)
with open(os.path.join(OUTPUT_DIR, 'annotations_swap.json'), "w") as outfile:
    json.dump(pseudo_anno_swap, outfile, cls=NpEncoder)
