import json
import numpy as np
import os
from sklearn.model_selection import StratifiedGroupKFold

annotation = "dataset/cleansing.json"
output_dir = 'dataset/json_folder'

with open(annotation) as f:
    data = json.load(f)

var = [(ann['image_id'], ann['category_id']) for ann in data['annotations']]
X = np.ones((len(data['annotations']), 1))
y = np.array([v[1] for v in var])
groups = np.array([v[0] for v in var])

cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=207)

for i,(train_idx, val_idx) in enumerate(cv.split(X, y, groups), start = 1) :
    train_data_img_ids = set([data['annotations'][idx]['image_id'] for idx in train_idx])
    train_data_imgs = [img for img in data['images'] for img_id in train_data_img_ids if img['id'] == img_id]
    train_data = {'images' : train_data_imgs,
                 'categories' : data['categories'],
                 'annotations': [data['annotations'][idx] for idx in train_idx],}
    
    val_data_img_ids = set([data['annotations'][idx]['image_id'] for idx in val_idx])
    val_data_imgs = [img for img in data['images'] for img_id in val_data_img_ids if img['id'] == img_id]
    val_data = {'images' : val_data_imgs,
                'categories' : data['categories'],
               'annotations': [data['annotations'][idx] for idx in val_idx],}
    
    train_filename = os.path.join(output_dir, f'train_fold_{i}.json')
    with open(train_filename, 'w') as f:
        json.dump(train_data, f, indent=4)
        
    val_filename = os.path.join(output_dir, f'val_fold_{i}.json')
    with open(val_filename, 'w') as f:
        json.dump(val_data, f, indent=4)