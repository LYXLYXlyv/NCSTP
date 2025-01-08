conv_kernel_size = 1
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
load_from = 'checkpoints/knet_s3_fcn_r50-d8_8x2_512x512_adamw_80k_ade20k_20220228_043751-abcab920.pth'
log_level = 'INFO'
log_processor = dict(by_epoch=False)
model = dict(
    auxiliary_head=dict(
        align_corners=False,
        channels=256,
        concat_input=False,
        dropout_ratio=0.1,
        in_channels=1024,
        in_index=2,
        loss_decode=dict(
            loss_weight=0.4, type='CrossEntropyLoss', use_sigmoid=False),
        norm_cfg=dict(requires_grad=True, type='BN'),
        num_classes=5,
        num_convs=1,
        type='FCNHead'),
    backbone=dict(
        contract_dilation=True,
        depth=50,
        dilations=(
            1,
            1,
            2,
            4,
        ),
        norm_cfg=dict(requires_grad=True, type='BN'),
        norm_eval=False,
        num_stages=4,
        out_indices=(
            0,
            1,
            2,
            3,
        ),
        strides=(
            1,
            2,
            1,
            1,
        ),
        style='pytorch',
        type='ResNetV1c'),
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
        kernel_generate_head=dict(
            align_corners=False,
            channels=512,
            dilations=(
                1,
                12,
                24,
                36,
            ),
            dropout_ratio=0.1,
            in_channels=2048,
            in_index=3,
            loss_decode=dict(
                loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=False),
            norm_cfg=dict(requires_grad=True, type='BN'),
            num_classes=5,
            type='ASPPHead'),
        kernel_update_head=[
            dict(
                conv_kernel_size=1,
                dropout=0.0,
                feat_transform_cfg=dict(
                    act_cfg=None, conv_cfg=dict(type='Conv2d')),
                feedforward_channels=2048,
                ffn_act_cfg=dict(inplace=True, type='ReLU'),
                in_channels=512,
                kernel_updator_cfg=dict(
                    act_cfg=dict(inplace=True, type='ReLU'),
                    feat_channels=256,
                    in_channels=256,
                    norm_cfg=dict(type='LN'),
                    out_channels=256,
                    type='KernelUpdator'),
                num_classes=5,
                num_ffn_fcs=2,
                num_heads=8,
                num_mask_fcs=1,
                out_channels=512,
                type='KernelUpdateHead',
                with_ffn=True),
            dict(
                conv_kernel_size=1,
                dropout=0.0,
                feat_transform_cfg=dict(
                    act_cfg=None, conv_cfg=dict(type='Conv2d')),
                feedforward_channels=2048,
                ffn_act_cfg=dict(inplace=True, type='ReLU'),
                in_channels=512,
                kernel_updator_cfg=dict(
                    act_cfg=dict(inplace=True, type='ReLU'),
                    feat_channels=256,
                    in_channels=256,
                    norm_cfg=dict(type='LN'),
                    out_channels=256,
                    type='KernelUpdator'),
                num_classes=5,
                num_ffn_fcs=2,
                num_heads=8,
                num_mask_fcs=1,
                out_channels=512,
                type='KernelUpdateHead',
                with_ffn=True),
            dict(
                conv_kernel_size=1,
                dropout=0.0,
                feat_transform_cfg=dict(
                    act_cfg=None, conv_cfg=dict(type='Conv2d')),
                feedforward_channels=2048,
                ffn_act_cfg=dict(inplace=True, type='ReLU'),
                in_channels=512,
                kernel_updator_cfg=dict(
                    act_cfg=dict(inplace=True, type='ReLU'),
                    feat_channels=256,
                    in_channels=256,
                    norm_cfg=dict(type='LN'),
                    out_channels=256,
                    type='KernelUpdator'),
                num_classes=5,
                num_ffn_fcs=2,
                num_heads=8,
                num_mask_fcs=1,
                out_channels=512,
                type='KernelUpdateHead',
                with_ffn=True),
        ],
        num_stages=3,
        type='IterativeDecodeHead'),
    pretrained='open-mmlab://resnet50_v1c',
    test_cfg=dict(mode='whole'),
    train_cfg=dict(),
    type='EncoderDecoder')
norm_cfg = dict(requires_grad=True, type='BN')
num_stages = 3
optim_wrapper = dict(
    clip_grad=dict(max_norm=1, norm_type=2),
    optimizer=dict(lr=0.0001, type='AdamW', weight_decay=0.0005),
    type='OptimWrapper')
optimizer = dict(lr=0.01, momentum=0.9, type='SGD', weight_decay=0.0005)
param_scheduler = [
    dict(
        begin=0, by_epoch=False, end=1000, start_factor=0.001,
        type='LinearLR'),
    dict(
        begin=1000,
        by_epoch=False,
        end=80000,
        milestones=[
            60000,
            72000,
        ],
        type='MultiStepLR'),
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
train_cfg = dict(max_iters=80000, type='IterBasedTrainLoop', val_interval=8000)
train_dataloader = dict(
    batch_size=8,
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