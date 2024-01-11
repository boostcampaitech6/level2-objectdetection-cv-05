import mmcv
from mmcv import Config
from mmdet.datasets import build_dataloader, build_dataset, replace_ImageToTensor
from mmdet.models import build_detector
from mmdet.apis import single_gpu_test
from mmcv.runner import load_checkpoint
import os
from mmcv.parallel import MMDataParallel
import pandas as pd
from pandas import DataFrame
from pycocotools.coco import COCO
import numpy as np
from mmdet.utils import get_device

CLASSES = (
    "General trash",
    "Paper",
    "Paper pack",
    "Metal",
    "Glass",
    "Plastic",
    "Styrofoam",
    "Plastic bag",
    "Battery",
    "Clothing",
)
# NAME = "convnext_xlarge"
NAME = "cascade_rcnn_r50_fpn_1x"
CV = "StratifiedGroupKFold"
FOLD = 1
# config_path = '/data/ephemeral/home/level2-objectdetection-cv-05/mmdetection/configs/convnext/cascade_rcnn_convnext_xlarge_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_2x_trash_aug.py'
config_path = "/data/ephemeral/home/level2-objectdetection-cv-05/mmdetection/configs/cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco.py"
cfg = Config.fromfile(config_path)

root = "/data/ephemeral/home/level2-objectdetection-cv-05/data/dataset/"

cfg.data.val.classes = CLASSES
cfg.data.val.img_prefix = root
cfg.data.val.ann_file = os.path.join(root, f"json_folder/{CV}/val_fold_{FOLD}.json")

cfg.data.test.classes = CLASSES
cfg.data.test.img_prefix = root
cfg.data.test.ann_file = os.path.join(root, "test.json")

cfg.val_pipeline = cfg.test_pipeline
cfg.test_pipeline = cfg.test_pipeline

cfg.data.val.pipeline = cfg.val_pipeline
cfg.data.test.pipeline = cfg.test_pipeline

cfg.data.samples_per_gpu = 16
cfg.data.workers_per_gpu = 4

cfg.seed = 207
cfg.gpu_ids = [0]
work_dir = f"./work_dirs/{NAME}_{CV}_{FOLD}"
cfg.work_dir = work_dir

cfg.optimizer_config.grad_clip = None
cfg.evaluation.classwise = True
cfg.evaluation.metric = "bbox"

# NUM_BBOX_HEAD = 3
NUM_BBOX_HEAD = len(cfg.model.roi_head.bbox_head)
for i in range(NUM_BBOX_HEAD):
    cfg.model.roi_head.bbox_head[i].num_classes = 10

cfg.checkpoint_config = dict(max_keep_ckpts=3, interval=1)


log_config = dict(_delete=True, interval=50, hooks=[dict(type="TextLoggerHook")])
cfg.log_config = log_config

cfg.model.train_cfg = None

# cfg.runner = dict(type='EpochBasedRunner', max_epochs=12)
cfg.device = get_device()

# build dataset & dataloader
dataset = build_dataset(cfg.data.test)
data_loader = build_dataloader(
    dataset,
    samples_per_gpu=1,
    workers_per_gpu=cfg.data.workers_per_gpu,
    dist=False,
    shuffle=False,
)

for i in [1, 2, 3, 4, 5]:
    # TODO: cfg for each fold #
    work_dir = (
        f"/data/ephemeral/home/level2-objectdetection-cv-05/work_dirs/{NAME}_{CV}_{i}"
    )
    cfg.work_dir = work_dir

    # checkpoint path
    checkpoint_path = os.path.join(cfg.work_dir, f"latest.pth")

    model = build_detector(cfg.model, test_cfg=cfg.get("test_cfg"))  # build detector
    checkpoint = load_checkpoint(
        model, checkpoint_path, map_location="cpu"
    )  # ckpt load
    ###########################
    model.CLASSES = dataset.CLASSES
    model = MMDataParallel(model.cuda(), device_ids=[0])

    output = single_gpu_test(model, data_loader, show_score_thr=0.05)  # output 계산

    # submission 양식에 맞게 output 후처리
    prediction_strings = []
    file_names = []
    coco = COCO(cfg.data.test.ann_file)
    img_ids = coco.getImgIds()

    class_num = 10
    for i, out in enumerate(output):
        prediction_string = ""
        image_info = coco.loadImgs(coco.getImgIds(imgIds=i))[0]
        for j in range(class_num):
            for o in out[j]:
                prediction_string += (
                    str(j)
                    + " "
                    + str(o[4])
                    + " "
                    + str(o[0])
                    + " "
                    + str(o[1])
                    + " "
                    + str(o[2])
                    + " "
                    + str(o[3])
                    + " "
                )

        prediction_strings.append(prediction_string)
        file_names.append(image_info["file_name"])

    submission = pd.DataFrame()
    submission["PredictionString"] = prediction_strings
    submission["image_id"] = file_names
    submission.to_csv(os.path.join(cfg.work_dir, f"submission_latest.csv"), index=None)
