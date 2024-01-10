from mmcv import Config
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector
from mmdet.utils import get_device

import os

CLASSES = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
        "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")
# NAME = "convnext_xlarge"
NAME = "cascade_rcnn_r50_fpn_1x"
# CV = 'StratifiedGroupKFold'
CV = 'MultilabelStratifiedKFold'
FOLD = 1
# config_path = '/data/ephemeral/home/level2-objectdetection-cv-05/mmdetection/configs/convnext/cascade_rcnn_convnext_xlarge_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_2x_trash_aug.py'
config_path = '/data/ephemeral/home/level2-objectdetection-cv-05/mmdetection/configs/cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco.py'
cfg = Config.fromfile(config_path)

root='/data/ephemeral/home/level2-objectdetection-cv-05/data/dataset/'

cfg.data.train.classes = CLASSES
cfg.data.train.img_prefix = root
cfg.data.train.ann_file = os.path.join(root, f'json_folder/{CV}/train_fold_{FOLD}.json')

cfg.data.val.classes = CLASSES
cfg.data.val.img_prefix = root
cfg.data.val.ann_file = os.path.join(root, f'json_folder/{CV}/val_fold_{FOLD}.json')

cfg.data.test.classes = CLASSES
cfg.data.test.img_prefix = root
cfg.data.test.ann_file = os.path.join(root, 'test.json')

cfg.train_pipeline = cfg.train_pipeline
cfg.val_pipeline = cfg.test_pipeline
cfg.test_pipeline = cfg.test_pipeline

cfg.data.train.pipeline = cfg.train_pipeline
cfg.data.val.pipeline = cfg.val_pipeline
cfg.data.test.pipeline = cfg.test_pipeline

cfg.data.samples_per_gpu = 16
cfg.data.workers_per_gpu = 4

cfg.seed = 207
cfg.gpu_ids = [0]
work_dir = f'./work_dirs/{NAME}_{CV}_{FOLD}'
cfg.work_dir = work_dir

cfg.optimizer_config.grad_clip = None
cfg.evaluation.classwise = True
cfg.evaluation.metric = 'bbox'

# NUM_BBOX_HEAD = 3
NUM_BBOX_HEAD = len(cfg.model.roi_head.bbox_head)
for i in range(NUM_BBOX_HEAD):
    cfg.model.roi_head.bbox_head[i].num_classes = 10

cfg.checkpoint_config = dict(max_keep_ckpts=3, interval=1, save_best='bbox_mAP')
    
# log_config = dict(
#     _delete=True,
#     interval=50,
#     hooks=[
#         dict(type='TextLoggerHook')]
#         )
# cfg.log_config = log_config

# cfg.runner = dict(type='EpochBasedRunner', max_epochs=12)
cfg.device = get_device()

for i in [1, 2, 3 ,4, 5]:
    # TODO: cfg for each fold #
    log_config = dict(
        _delete=True,
        interval=50,
        hooks=[
            dict(type='TextLoggerHook'),
            dict(type='MMDetWandbHook',
                init_kwargs=dict(
                    project = f'mmdetection_{NAME}_{CV}',
                    # entity = 'ENTITY 이름',
                    name = f'fold_{i}'
                ),
                interval=10,
                log_checkpoint=False,
                log_checkpoint_metadata=True,
                # num_eval_images=2272,
                num_eval_images=100,
                bbox_score_thr=0.3)]
            )
    cfg.log_config = log_config

    cfg.data.train.ann_file = os.path.join(root, f'json_folder/{CV}/train_fold_{FOLD}.json')
    cfg.data.val.ann_file = os.path.join(root, f'json_folder/{CV}/val_fold_{FOLD}.json')
    # datasets = [build_dataset(cfg.data.train)]
    datasets = [build_dataset(cfg.data.train), build_dataset(cfg.data.val)]

    work_dir = f'./work_dirs/{NAME}_{CV}_{i}'
    cfg.work_dir = work_dir
    ###########################

    print(datasets[0])

    model = build_detector(cfg.model)
    model.init_weights()

    train_detector(model, datasets[0], cfg, distributed=False, validate=True)