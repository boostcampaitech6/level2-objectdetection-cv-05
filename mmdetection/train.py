from mmcv import Config
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector
from mmdet.utils import get_device

import os

CLASSES = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
        "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")
NAME = "convnext_xlarge"
config_path = '/data/ephemeral/home/level2-objectdetection-cv-05/mmdetection/configs/convnext/cascade_rcnn_convnext_xlarge_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_2x_trash_aug.py'
cfg = Config.fromfile(config_path)

root='../../dataset/'

cfg.data.train.classes = CLASSES
cfg.data.train.img_prefix = root
cfg.data.train.ann_file = os.path.join(root, 'json_folder/train_fold_1.json')

cfg.data.val.classes = CLASSES
cfg.data.val.img_prefix = root
cfg.data.val.ann_file = os.path.join(root, 'json_folder/val_fold_1.json')

cfg.data.test.classes = CLASSES
cfg.data.test.img_prefix = root
cfg.data.test.ann_file = os.path.join(root, 'test.json')

cfg.train_pipeline = cfg.train_pipeline
cfg.val_pipeline = cfg.test_pipeline
cfg.test_pipeline = cfg.test_pipeline

cfg.data.train.pipeline = cfg.train_pipeline
cfg.data.val.pipeline = cfg.val_pipeline
cfg.data.test.pipeline = cfg.test_pipeline

cfg.data.samples_per_gpu = 1
cfg.data.workers_per_gpu = 4

cfg.seed = 207
cfg.gpu_ids = [0]
work_dir = f'./work_dirs/{NAME}'
# if not os.path.exists(work_dir):
#     os.mkdir(work_dir)
    
cfg.work_dir = work_dir
cfg.optimizer_config.grad_clip = None
cfg.evaluation.classwise = True
cfg.evaluation.metric = 'bbox'

NUM_BBOX_HEAD = 3
for i in range(NUM_BBOX_HEAD):
    cfg.model.roi_head.bbox_head[i].num_classes = 10

cfg.checkpoint_config = dict(max_keep_ckpts=3, interval=1)

    
log_config = dict(
    _delete=True,
    interval=50,
    hooks=[
        dict(type='TextLoggerHook')
        ])
cfg.log_config = log_config


# cfg.runner = dict(type='EpochBasedRunner', max_epochs=12)
cfg.device = get_device()


datasets = [build_dataset(cfg.data.train)]

print(datasets[0])

model = build_detector(cfg.model)
model.init_weights()

train_detector(model, datasets[0], cfg, distributed=False, validate=True)