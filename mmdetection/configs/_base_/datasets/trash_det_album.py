dataset_type = "CocoDataset"
data_root = "/data/ephemeral/home/dataset"

""" todo """
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)

album = [
    dict(
        type="OneOf",
        transforms=[dict(type="Flip", p=1.0), dict(type="RandomRotate90", p=1.0)],
        p=0.5,
    ),
    dict(
        type="OneOf",
        transforms=[
            dict(
                type="RandomBrightnessContrast",
                brightness_limit=(-0.1, 0.15),
                contrast_limit=(-0.1, 0.15),
                p=1.0,
            ),
            dict(
                type="HueSaturationValue",
                hue_shift_limit=15,
                sat_shift_limit=25,
                val_shift_limit=10,
                p=0.5,
            ),
        ],
        p=0.5,
    ),
    dict(type="GaussNoise", var_limit=(20, 100), p=0.3),
    dict(
        type="OneOf",
        transforms=[
            dict(type="Blur", p=1.0),
            dict(type="GaussianBlur", p=1.0),
            dict(type="MedianBlur", blur_limit=5, p=1.0),
            dict(type="MotionBlur", p=1.0),
        ],
        p=0.1,
    ),
]

# augmentation strategy originates from DETR / Sparse RCNN
train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(type="RandomFlip", flip_ratio=0.5),
    dict(
        type="Resize",
        img_scale=[
            (480, 1024),
            (512, 1024),
            (544, 1024),
            (576, 1024),
            (608, 1024),
            (640, 1024),
            (672, 1024),
            (704, 1024),
            (736, 1024),
            (768, 1024),
            (800, 1024),
            (1024, 1024),
        ],
        multiscale_mode="value",
        keep_ratio=True,
    ),
    dict(
        type="Albu",
        transforms=album,
        bbox_params=dict(
            type="BboxParams",
            format="pascal_voc",
            label_fields=["gt_labels"],
            min_visibility=0.0,
            filter_lost_elements=True,
        ),
        keymap={"img": "image", "gt_bboxes": "bboxes"},
        update_pad_shape=False,
        skip_img_without_anno=True,
    ),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="Pad", size_divisor=32),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "gt_bboxes", "gt_labels"]),
]

test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(1024, 1024),
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="Pad", size_divisor=32),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]

data = dict(
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline),
    samples_per_gpu=2,
)

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file=data_root + "/json_folder/train_fold_1.json",
        img_prefix=data_root,
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset_type,
        ann_file=data_root + "/json_folder/val_fold_1.json",
        img_prefix=data_root,
        pipeline=test_pipeline,
    ),
    test=dict(
        type=dataset_type,
        ann_file=data_root + "/test.json",
        img_prefix=data_root,
        pipeline=test_pipeline,
    ),
)

evaluation = dict(interval=1, metric="bbox")
