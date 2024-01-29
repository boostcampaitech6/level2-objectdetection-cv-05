from mmcv import Config
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector
from mmdet.utils import get_device

import os
import argparse

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


def train(args):
    NAME = args.name
    config_path = args.config
    cfg = Config.fromfile(config_path)
    root = args.root

    cfg.data.train.classes = CLASSES
    cfg.data.train.img_prefix = root
    cfg.data.train.ann_file = os.path.join(root, args.train_annotation)

    cfg.data.val.classes = CLASSES
    cfg.data.val.img_prefix = root
    cfg.data.val.ann_file = os.path.join(root, args.valid_annotation)

    cfg.data.test.classes = CLASSES
    cfg.data.test.img_prefix = root
    cfg.data.test.ann_file = os.path.join(root, "test.json")

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
    work_dir = f"./work_dirs/{NAME}"

    cfg.work_dir = work_dir
    cfg.optimizer_config.grad_clip = None
    cfg.evaluation.classwise = True
    cfg.evaluation.metric = "bbox"

    cfg.checkpoint_config = dict(max_keep_ckpts=3, interval=1)
    log_config = dict(
        _delete=True,
        interval=50,
        hooks=[
            dict(type="TextLoggerHook"),
            dict(
                type="MMDetWandbHook",
                init_kwargs=dict(
                    project=f"mmdetection",
                    # entity = 'ENTITY 이름',
                    name=f"{NAME}",
                ),
                interval=10,
                log_checkpoint=False,
                log_checkpoint_metadata=True,
                num_eval_images=100,
                bbox_score_thr=0.3,
            ),
        ],
    )
    cfg.log_config = log_config
    cfg.device = get_device()

    datasets = [build_dataset(cfg.data.train), build_dataset(cfg.data.val)]

    print(datasets[0])

    model = build_detector(cfg.model)
    model.init_weights()

    train_detector(model, datasets[0], cfg, distributed=False, validate=True)


if __name__ == "__main__":
    # ----- Parser -----
    parser = argparse.ArgumentParser(
        description="Script to train mmdetection detector."
    )
    parser.add_argument(
        "-n", "--name", type=str, default="convnextlarge", help="experiment name"
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="/data/ephemeral/home/level2-objectdetection-cv-05/mmdetection/configs/convnext/cascade_rcnn_convnext_xlarge_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_2x_trash_aug.py",
        help="path to config.py",
    )
    parser.add_argument(
        "-r",
        "--root",
        type=str,
        default="/data/ephemeral/home/level2-objectdetection-cv-05/dataset",
        help="path to dataset folder",
    )
    parser.add_argument(
        "-t",
        "--train_annotation",
        type=str,
        default="json_folder/StratifiedGroupKFold/train_fold_1.json",
        help="path to train.json",
    )
    parser.add_argument(
        "-v",
        "--valid_annotation",
        type=str,
        default="json_folder/StratifiedGroupKFold/val_fold_1.json",
        help="path to valid.json",
    )

    args = parser.parse_args()
    print(args)
    train(args)
