crop_size = (
    512,
    1024,
)
data_preprocessor = dict(
    bgr_to_rgb=True,
    mean=[
        123.675,
        116.28,
        103.53,
    ],
    pad_val=0,
    seg_pad_val=255,
    size=(
        512,
        1024,
    ),
    std=[
        58.395,
        57.12,
        57.375,
    ],
    type='SegDataPreProcessor')
data_root = 'data/SNTCR_dataset/'
dataset_type = 'SNTCRDataset'
default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval=8000, type='CheckpointHook'),
    logger=dict(interval=50, log_metric_by_epoch=False, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='SegVisualizationHook'))
default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
img_ratios = [
    0.5,
    0.75,
    1.0,
    1.25,
    1.5,
    1.75,
]
launcher = 'none'
load_from = 'checkpoints/fast_scnn_lr0.12_8x4_160k_cityscapes_20210630_164853-0cec9937.pth'
log_level = 'INFO'
log_processor = dict(by_epoch=False)
model = dict(
    auxiliary_head=[
        dict(
            align_corners=False,
            channels=32,
            concat_input=False,
            in_channels=128,
            in_index=-2,
            loss_decode=dict(
                loss_weight=0.4, type='CrossEntropyLoss', use_sigmoid=True),
            norm_cfg=dict(momentum=0.01, requires_grad=True, type='BN'),
            num_classes=5,
            num_convs=1,
            type='FCNHead'),
        dict(
            align_corners=False,
            channels=32,
            concat_input=False,
            in_channels=64,
            in_index=-3,
            loss_decode=dict(
                loss_weight=0.4, type='CrossEntropyLoss', use_sigmoid=True),
            norm_cfg=dict(momentum=0.01, requires_grad=True, type='BN'),
            num_classes=5,
            num_convs=1,
            type='FCNHead'),
    ],
    backbone=dict(
        align_corners=False,
        downsample_dw_channels=(
            32,
            48,
        ),
        fusion_out_channels=128,
        global_block_channels=(
            64,
            96,
            128,
        ),
        global_block_strides=(
            2,
            2,
            1,
        ),
        global_in_channels=64,
        global_out_channels=128,
        higher_in_channels=64,
        lower_in_channels=128,
        norm_cfg=dict(momentum=0.01, requires_grad=True, type='SyncBN'),
        out_indices=(
            0,
            1,
            2,
        ),
        type='FastSCNN'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_val=0,
        seg_pad_val=255,
        size=(
            512,
            1024,
        ),
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='SegDataPreProcessor'),
    decode_head=dict(
        align_corners=False,
        channels=128,
        concat_input=False,
        in_channels=128,
        in_index=-1,
        loss_decode=dict(
            loss_weight=1, type='CrossEntropyLoss', use_sigmoid=True),
        norm_cfg=dict(momentum=0.01, requires_grad=True, type='SyncBN'),
        num_classes=19,
        type='DepthwiseSeparableFCNHead'),
    test_cfg=dict(mode='whole'),
    train_cfg=dict(),
    type='EncoderDecoder')
norm_cfg = dict(momentum=0.01, requires_grad=True, type='BN')
optim_wrapper = dict(
    clip_grad=None,
    optimizer=dict(lr=0.12, momentum=0.9, type='SGD', weight_decay=4e-05),
    type='OptimWrapper')
optimizer = dict(lr=0.12, momentum=0.9, type='SGD', weight_decay=4e-05)
param_scheduler = [
    dict(
        begin=0,
        by_epoch=False,
        end=160000,
        eta_min=0.0001,
        power=0.9,
        type='PolyLR'),
]
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_prefix=dict(img_path='img_dir/val', seg_map_path='ann_dir/val'),
        data_root='data/SNTCR_dataset/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                2048,
                1024,
            ), type='Resize'),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        type='SNTCRDataset'),
    num_workers=0,
    persistent_workers=False,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    iou_metrics=[
        'mIoU',
    ], type='IoUMetric')
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(keep_ratio=True, scale=(
        2048,
        1024,
    ), type='Resize'),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs'),
]
train_cfg = dict(
    max_iters=40000, type='IterBasedTrainLoop', val_interval=4000)
train_dataloader = dict(
    batch_size=16,
    dataset=dict(
        data_prefix=dict(
            img_path='img_dir/train', seg_map_path='ann_dir/train'),
        data_root='data/SNTCR_dataset/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(
                keep_ratio=True,
                ratio_range=(
                    0.5,
                    2.0,
                ),
                scale=(
                    2048,
                    1024,
                ),
                type='RandomResize'),
            dict(
                cat_max_ratio=0.75, crop_size=(
                    512,
                    1024,
                ), type='RandomCrop'),
            dict(prob=0.5, type='RandomFlip'),
            dict(type='PhotoMetricDistortion'),
            dict(type='PackSegInputs'),
        ],
        type='SNTCRDataset'),
    num_workers=0,
    persistent_workers=False,
    sampler=dict(shuffle=True, type='InfiniteSampler'))
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        keep_ratio=True,
        ratio_range=(
            0.5,
            2.0,
        ),
        scale=(
            2048,
            1024,
        ),
        type='RandomResize'),
    dict(cat_max_ratio=0.75, crop_size=(
        512,
        1024,
    ), type='RandomCrop'),
    dict(prob=0.5, type='RandomFlip'),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_prefix=dict(img_path='img_dir/val', seg_map_path='ann_dir/val'),
        data_root='data/SNTCR_dataset/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                2048,
                1024,
            ), type='Resize'),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        type='SNTCRDataset'),
    num_workers=0,
    persistent_workers=False,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    iou_metrics=[
        'mIoU',
    ], type='IoUMetric')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='SegLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])