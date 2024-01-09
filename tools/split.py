import json
import numpy as np
import os
from sklearn.model_selection import StratifiedGroupKFold
try:
    from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
except ImportError:
    raise ImportError('Please run `pip install iterative-stratification` to install '
                      '3rd party package iterstrat.')
from collections import Counter

# TODO #
annotation = "../dataset/train.json"
output_dir = '../dataset/json_folder'
METHOD = 1
########
methods = {
    0: "StratifiedGroupKFold",
    1: "MultilabelStratifiedKFold",
}

with open(annotation) as f:
    data = json.load(f)

var = [(ann['image_id'], ann['category_id'], ann['bbox']) for ann in data['annotations']]

os.makedirs(os.path.join(output_dir, methods[METHOD]), exist_ok=True)

if METHOD == 0:
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=207)

    X = np.ones((len(data['annotations']), 1))
    y = np.array([v[1] for v in var])
    groups = np.array([v[0] for v in var])  # image_id
                                            # groups: 같은 image 내 box

    for i,(train_idx, val_idx) in enumerate(cv.split(X, y, groups), start = 1) :
        train_data_img_ids = set([data['annotations'][idx]['image_id'] for idx in train_idx])
        train_data_imgs = [data['images'][idx] for idx in train_data_img_ids]
        train_data = {'images' : train_data_imgs,
                    'categories' : data['categories'],
                    'annotations': [data['annotations'][idx] for idx in train_idx],}
        
        val_data_img_ids = set([data['annotations'][idx]['image_id'] for idx in val_idx])
        val_data_imgs = [data['images'][idx] for idx in val_data_img_ids]
        val_data = {'images' : val_data_imgs,
                    'categories' : data['categories'],
                'annotations': [data['annotations'][idx] for idx in val_idx],}
        
        train_filename = os.path.join(output_dir, methods[METHOD], f'train_fold_{i}.json')
        with open(train_filename, 'w') as f:
            json.dump(train_data, f, indent=4)
            
        val_filename = os.path.join(output_dir, methods[METHOD], f'val_fold_{i}.json')
        with open(val_filename, 'w') as f:
            json.dump(val_data, f, indent=4)

# ref: https://github.com/amirassov/kaggle-global-wheat-detection/blob/master/gwd/split_folds.py
elif METHOD == 1:
    cv = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=207)

    X = np.ones((len(data['annotations']), 1))
    groups = np.array([v[0] for v in var])      # image_id
    nums = dict(Counter([v[0] for v in var]))   # num_of_bboxes
    c = np.array([v[1] for v in var])           # category_id
    b = np.array([v[2] for v in var])           # bbox

    ms = {}
    ms['image_id'] = groups
    ms['categories'] = c
    ms['median_area'] = np.apply_along_axis(lambda x: np.sqrt(x[-1] * x[-2]), 1, b)
    ms['num_of_bboxes'] = np.array([nums[x] for x in [v[0] for v in var]])

    for i, (train_idx, val_idx) in enumerate(cv.split(X, np.column_stack((ms['categories'], ms['median_area'], ms['num_of_bboxes']))), start=1):
        train_data_img_ids = set([data['annotations'][idx]['image_id'] for idx in train_idx])
        train_data_imgs = [data['images'][idx] for idx in train_data_img_ids]
        train_data = {'images' : train_data_imgs,
                    'categories' : data['categories'],
                    'annotations': [data['annotations'][idx] for idx in train_idx],}
        
        val_data_img_ids = set([data['annotations'][idx]['image_id'] for idx in val_idx])
        val_data_imgs = [data['images'][idx] for idx in val_data_img_ids]
        val_data = {'images' : val_data_imgs,
                    'categories' : data['categories'],
                'annotations': [data['annotations'][idx] for idx in val_idx],}
        
        train_filename = os.path.join(output_dir, methods[METHOD], f'train_fold_{i}.json')
        with open(train_filename, 'w') as f:
            json.dump(train_data, f, indent=4)
            
        val_filename = os.path.join(output_dir, methods[METHOD], f'val_fold_{i}.json')
        with open(val_filename, 'w') as f:
            json.dump(val_data, f, indent=4)