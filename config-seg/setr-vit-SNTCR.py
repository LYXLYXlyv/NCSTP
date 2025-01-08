backbone_norm_cfg = dict(eps=1e-06, requires_grad=True, type='LN')
crop_size = (
    512,
    512,
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
        512,
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
    checkpoint=dict(by_epoch=False, interval=16000, type='CheckpointHook'),
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
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=False)
model = dict(
    auxiliary_head=[
        dict(
            act_cfg=dict(type='ReLU'),
            align_corners=False,
            channels=256,
            concat_input=False,
            dropout_ratio=0,
            in_channels=256,
            in_index=0,
            kernel_size=1,
            loss_decode=dict(
                loss_weight=0.4, type='CrossEntropyLoss', use_sigmoid=False),
            norm_cfg=dict(requires_grad=True, type='BN'),
            num_classes=5,
            num_convs=0,
            type='FCNHead'),
        dict(
            act_cfg=dict(type='ReLU'),
            align_corners=False,
            channels=256,
            concat_input=False,
            dropout_ratio=0,
            in_channels=256,
            in_index=1,
            kernel_size=1,
            loss_decode=dict(
                loss_weight=0.4, type='CrossEntropyLoss', use_sigmoid=False),
            norm_cfg=dict(requires_grad=True, type='BN'),
            num_classes=5,
            num_convs=0,
            type='FCNHead'),
        dict(
            act_cfg=dict(type='ReLU'),
            align_corners=False,
            channels=256,
            concat_input=False,
            dropout_ratio=0,
            in_channels=256,
            in_index=2,
            kernel_size=1,
            loss_decode=dict(
                loss_weight=0.4, type='CrossEntropyLoss', use_sigmoid=False),
            norm_cfg=dict(requires_grad=True, type='BN'),
            num_classes=5,
            num_convs=0,
            type='FCNHead'),
        dict(
            act_cfg=dict(type='ReLU'),
            align_corners=False,
            channels=256,
            concat_input=False,
            dropout_ratio=0,
            in_channels=256,
            in_index=3,
            kernel_size=1,
            loss_decode=dict(
                loss_weight=0.4, type='CrossEntropyLoss', use_sigmoid=False),
            norm_cfg=dict(requires_grad=True, type='BN'),
            num_classes=5,
            num_convs=0,
            type='FCNHead'),
    ],
    backbone=dict(
        drop_rate=0.0,
        embed_dims=1024,
        img_size=(
            512,
            512,
        ),
        in_channels=3,
        init_cfg=dict(
            checkpoint='pretrain/vit_large_p16.pth', type='Pretrained'),
        interpolate_mode='bilinear',
        norm_cfg=dict(eps=1e-06, requires_grad=True, type='LN'),
        num_heads=16,
        num_layers=24,
        out_indices=(
            5,
            11,
            17,
            23,
        ),
        patch_size=16,
        type='VisionTransformer',
        with_cls_token=False),
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
            512,
        ),
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='SegDataPreProcessor'),
    decode_head=dict(
        align_corners=False,
        channels=512,
        dropout_ratio=0,
        in_channels=(
            256,
            256,
            256,
            256,
        ),
        in_index=(
            0,
            1,
            2,
            3,
        ),
        loss_decode=dict(
            loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=False),
        mla_channels=128,
        norm_cfg=dict(requires_grad=True, type='BN'),
        num_classes=5,
        type='SETRMLAHead'),
    neck=dict(
        act_cfg=dict(type='ReLU'),
        in_channels=[
            1024,
            1024,
            1024,
            1024,
        ],
        norm_cfg=dict(requires_grad=True, type='BN'),
        out_channels=256,
        type='MLANeck'),
    pretrained=None,
    test_cfg=dict(crop_size=(
        512,
        512,
    ), mode='slide', stride=(
        341,
        341,
    )),
    train_cfg=dict(),
    type='EncoderDecoder')
norm_cfg = dict(requires_grad=True, type='BN')
optim_wrapper = dict(
    clip_grad=None,
    optimizer=dict(lr=0.001, momentum=0.9, type='SGD', weight_decay=0.0),
    paramwise_cfg=dict(custom_keys=dict(head=dict(lr_mult=10.0))),
    type='OptimWrapper')
optimizer = dict(lr=0.001, momentum=0.9, type='SGD', weight_decay=0.0)
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
        data_prefix=dict(
            img_path='img_dir/val', seg_map_path='ann_dir/val'),
        data_root=data_root,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                2048,
                512,
            ), type='Resize'),
            dict(reduce_zero_label=False, type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        type=dataset_type),
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
        512,
    ), type='Resize'),
    dict(reduce_zero_label=False, type='LoadAnnotations'),
    dict(type='PackSegInputs'),
]
train_cfg = dict(
    max_iters=160000, type='IterBasedTrainLoop', val_interval=16000)
train_dataloader = dict(
    batch_size=4,
    dataset=dict(
        data_prefix=dict(
            img_path='img_dir/train', seg_map_path='ann_dir/train'),
        data_root=data_root,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(reduce_zero_label=False, type='LoadAnnotations'),
            dict(
                keep_ratio=True,
                ratio_range=(
                    0.5,
                    2.0,
                ),
                scale=(
                    2048,
                    512,
                ),
                type='RandomResize'),
            dict(
                cat_max_ratio=0.75, crop_size=(
                    512,
                    512,
                ), type='RandomCrop'),
            dict(prob=0.5, type='RandomFlip'),
            dict(type='PhotoMetricDistortion'),
            dict(type='PackSegInputs'),
        ],
        type=dataset_type),
    num_workers=0,
    persistent_workers=False,
    sampler=dict(shuffle=True, type='InfiniteSampler'))
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(reduce_zero_label=False, type='LoadAnnotations'),
    dict(
        keep_ratio=True,
        ratio_range=(
            0.5,
            2.0,
        ),
        scale=(
            2048,
            512,
        ),
        type='RandomResize'),
    dict(cat_max_ratio=0.75, crop_size=(
        512,
        512,
    ), type='RandomCrop'),
    dict(prob=0.5, type='RandomFlip'),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_prefix=dict(
            img_path='img_dir/val', seg_map_path='ann_dir/val'),
        data_root=data_root,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                2048,
                512,
            ), type='Resize'),
            dict(reduce_zero_label=False, type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        type=dataset_type),
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