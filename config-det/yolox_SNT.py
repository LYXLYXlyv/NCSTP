auto_scale_lr = dict(base_batch_size=64, enable=False)
backend_args = None
base_lr = 0.01
custom_hooks = [
    dict(num_last_epochs=15, priority=48, type='YOLOXModeSwitchHook'),
    dict(priority=48, type='SyncNormHook'),
    dict(
        ema_type='ExpMomentumEMA',
        momentum=0.0001,
        priority=49,
        type='EMAHook',
        update_buffers=True),
]
data_root = 'data/SNT_coco/'
dataset_type = 'SNTDataset'
default_hooks = dict(
    checkpoint=dict(interval=6, type='CheckpointHook'),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='DetVisualizationHook'))
default_scope = 'mmdet'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
img_scale = (
    640,
    640,
)
img_scales = [
    (
        640,
        640,
    ),
    (
        320,
        320,
    ),
    (
        960,
        960,
    ),
]
interval = 10
load_from = 'pth/yolox_s_8x8_300e_coco_20211121_095711-4592a793.pth'
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
max_epochs = 300
model = dict(
    backbone=dict(
        act_cfg=dict(type='Swish'),
        deepen_factor=0.33,
        norm_cfg=dict(eps=0.001, momentum=0.03, type='BN'),
        out_indices=(
            2,
            3,
            4,
        ),
        spp_kernal_sizes=(
            5,
            9,
            13,
        ),
        type='CSPDarknet',
        use_depthwise=False,
        widen_factor=0.5),
    bbox_head=dict(
        act_cfg=dict(type='Swish'),
        feat_channels=128,
        in_channels=128,
        loss_bbox=dict(
            eps=1e-16,
            loss_weight=5.0,
            mode='square',
            reduction='sum',
            type='IoULoss'),
        loss_cls=dict(
            loss_weight=1.0,
            reduction='sum',
            type='CrossEntropyLoss',
            use_sigmoid=True),
        loss_l1=dict(loss_weight=1.0, reduction='sum', type='L1Loss'),
        loss_obj=dict(
            loss_weight=1.0,
            reduction='sum',
            type='CrossEntropyLoss',
            use_sigmoid=True),
        norm_cfg=dict(eps=0.001, momentum=0.03, type='BN'),
        num_classes=12,
        stacked_convs=2,
        strides=(
            8,
            16,
            32,
        ),
        type='YOLOXHead',
        use_depthwise=False),
    data_preprocessor=dict(
        batch_augments=[
            dict(
                interval=10,
                random_size_range=(
                    480,
                    800,
                ),
                size_divisor=32,
                type='BatchSyncRandomResize'),
        ],
        pad_size_divisor=32,
        type='DetDataPreprocessor'),
    neck=dict(
        act_cfg=dict(type='Swish'),
        in_channels=[
            128,
            256,
            512,
        ],
        norm_cfg=dict(eps=0.001, momentum=0.03, type='BN'),
        num_csp_blocks=1,
        out_channels=128,
        type='YOLOXPAFPN',
        upsample_cfg=dict(mode='nearest', scale_factor=2),
        use_depthwise=False),
    test_cfg=dict(nms=dict(iou_threshold=0.65, type='nms'), score_thr=0.01),
    train_cfg=dict(assigner=dict(center_radius=2.5, type='SimOTAAssigner')),
    type='YOLOX')
num_last_epochs = 15
optim_wrapper = dict(
    optimizer=dict(
        lr=0.01, momentum=0.9, nesterov=True, type='SGD', weight_decay=0.0005),
    paramwise_cfg=dict(bias_decay_mult=0.0, norm_decay_mult=0.0),
    type='OptimWrapper')
param_scheduler = [
    dict(
        begin=0,
        by_epoch=True,
        convert_to_iter_based=True,
        end=5,
        type='mmdet.QuadraticWarmupLR'),
    dict(
        T_max=285,
        begin=5,
        by_epoch=True,
        convert_to_iter_based=True,
        end=285,
        eta_min=0.0005,
        type='CosineAnnealingLR'),
    dict(begin=285, by_epoch=True, end=300, factor=1, type='ConstantLR'),
]
resume = False
metainfo = {
        'classes':
        ("Cubesat","Cylindrical_satellite", "GEO_communication_satellite", "LEO_communication_satellite",
         "Navigation_satellite", "SAR_satellite", "Single_panel_Earth_observation_satellite", "Symmetrical_Earth_Observation_Satellite",
         "Space_Probe", "Space_Telescope", "spacedebris", "spacerock"),
        # palette is a list of color tuples, which is used for visualization.
        'palette':
        [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
         (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30),
         (100, 170, 30), (220, 220, 0)]
    }
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=8,
    dataset=dict(
        ann_file='annotations/test.json',
        backend_args=None,
        data_prefix=dict(img='test/'),
        data_root=data_root,
        metainfo=metainfo,
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                640,
                640,
            ), type='Resize'),
            dict(
                pad_to_square=True,
                pad_val=dict(img=(
                    114.0,
                    114.0,
                    114.0,
                )),
                type='Pad'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetInputs'),
        ],
        test_mode=True,
        type=dataset_type),
    drop_last=False,
    num_workers=0,
    persistent_workers=False,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    ann_file=data_root+'annotations/test.json',
    backend_args=None,
    metric='bbox',
    type='CocoMetric')
test_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(keep_ratio=True, scale=(
        640,
        640,
    ), type='Resize'),
    dict(
        pad_to_square=True,
        pad_val=dict(img=(
            114.0,
            114.0,
            114.0,
        )),
        type='Pad'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
        ),
        type='PackDetInputs'),
]
train_cfg = dict(max_epochs=12, type='EpochBasedTrainLoop', val_interval=1)
train_dataloader = dict(
    batch_size=16,
    dataset=dict(
        dataset=dict(
        ann_file='annotations/train.json',
        backend_args=None,
        data_prefix=dict(img='train/'),
        data_root='data/SNT_coco/',
        metainfo=metainfo,
            filter_cfg=dict(filter_empty_gt=False, min_size=32),
            pipeline=[
                dict(backend_args=None, type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True),
            ],
            type=dataset_type),
        pipeline=[
            dict(img_scale=(
                640,
                640,
            ), pad_val=114.0, type='Mosaic'),
            dict(
                border=(
                    -320,
                    -320,
                ),
                scaling_ratio_range=(
                    0.1,
                    2,
                ),
                type='RandomAffine'),
            dict(
                img_scale=(
                    640,
                    640,
                ),
                pad_val=114.0,
                ratio_range=(
                    0.8,
                    1.6,
                ),
                type='MixUp'),
            dict(type='YOLOXHSVRandomAug'),
            dict(prob=0.5, type='RandomFlip'),
            dict(keep_ratio=True, scale=(
                640,
                640,
            ), type='Resize'),
            dict(
                pad_to_square=True,
                pad_val=dict(img=(
                    114.0,
                    114.0,
                    114.0,
                )),
                type='Pad'),
            dict(
                keep_empty=False,
                min_gt_bbox_wh=(
                    1,
                    1,
                ),
                type='FilterAnnotations'),
            dict(type='PackDetInputs'),
        ],
        type='MultiImageMixDataset'),
    num_workers=0,
    persistent_workers=False,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_dataset = dict(
    dataset=dict(
        ann_file='annotations/train.json',
        backend_args=None,
        data_prefix=dict(img='train/'),
        data_root='data/SNT_coco/',
        metainfo=metainfo,
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
        ],
        type=dataset_type),
    pipeline=[
        dict(img_scale=(
            640,
            640,
        ), pad_val=114.0, type='Mosaic'),
        dict(
            border=(
                -320,
                -320,
            ),
            scaling_ratio_range=(
                0.1,
                2,
            ),
            type='RandomAffine'),
        dict(
            img_scale=(
                640,
                640,
            ),
            pad_val=114.0,
            ratio_range=(
                0.8,
                1.6,
            ),
            type='MixUp'),
        dict(type='YOLOXHSVRandomAug'),
        dict(prob=0.5, type='RandomFlip'),
        dict(keep_ratio=True, scale=(
            640,
            640,
        ), type='Resize'),
        dict(
            pad_to_square=True,
            pad_val=dict(img=(
                114.0,
                114.0,
                114.0,
            )),
            type='Pad'),
        dict(
            keep_empty=False,
            min_gt_bbox_wh=(
                1,
                1,
            ),
            type='FilterAnnotations'),
        dict(type='PackDetInputs'),
    ],
    type='MultiImageMixDataset')
train_pipeline = [
    dict(img_scale=(
        640,
        640,
    ), pad_val=114.0, type='Mosaic'),
    dict(
        border=(
            -320,
            -320,
        ),
        scaling_ratio_range=(
            0.1,
            2,
        ),
        type='RandomAffine'),
    dict(
        img_scale=(
            640,
            640,
        ),
        pad_val=114.0,
        ratio_range=(
            0.8,
            1.6,
        ),
        type='MixUp'),
    dict(type='YOLOXHSVRandomAug'),
    dict(prob=0.5, type='RandomFlip'),
    dict(keep_ratio=True, scale=(
        640,
        640,
    ), type='Resize'),
    dict(
        pad_to_square=True,
        pad_val=dict(img=(
            114.0,
            114.0,
            114.0,
        )),
        type='Pad'),
    dict(keep_empty=False, min_gt_bbox_wh=(
        1,
        1,
    ), type='FilterAnnotations'),
    dict(type='PackDetInputs'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=8,
    dataset=dict(
         ann_file='annotations/val.json',
        backend_args=None,
        data_prefix=dict(img='val/'),
        data_root=data_root,
        metainfo=metainfo,
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                640,
                640,
            ), type='Resize'),
            dict(
                pad_to_square=True,
                pad_val=dict(img=(
                    114.0,
                    114.0,
                    114.0,
                )),
                type='Pad'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetInputs'),
        ],
        test_mode=True,
        type=dataset_type),
    drop_last=False,
    num_workers=0,
    persistent_workers=False,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    ann_file=data_root+'annotations/val.json',
    backend_args=None,
    metric='bbox',
    type='CocoMetric')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])