crop_size = (
    480,
    480,
)
data_preprocessor = dict(
    bgr_to_rgb=True,
    mean=[
        127.5,
        127.5,
        127.5,
    ],
    pad_val=0,
    seg_pad_val=0,
    std=[
        127.5,
        127.5,
        127.5,
    ],
    type='SegDataPreProcessor')
data_root = 'data/SNTCR_dataset/'
dataset_type = 'SNTCRDataset'
default_hooks = dict(
    checkpoint=dict(
        by_epoch=False, interval=80000, save_best='mIoU',
        type='CheckpointHook'),
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
find_unused_parameters = True
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=False)
model = dict(
    backbone=dict(
        class_embed_path=
        'https://download.openmmlab.com/mmsegmentation/v0.5/vpd/nyu_class_embeddings.pth',
        class_embed_select=True,
        diffusion_cfg=dict(
            base_learning_rate=0.0001,
            checkpoint=
            'https://download.openmmlab.com/mmsegmentation/v0.5/vpd/stable_diffusion_v1-5_pretrain_third_party.pth',
            params=dict(
                channels=4,
                cond_stage_config=dict(
                    target='ldm.modules.encoders.modules.AbstractEncoder'),
                cond_stage_key='txt',
                cond_stage_trainable=False,
                conditioning_key='crossattn',
                first_stage_config=dict(
                    params=dict(
                        ddconfig=dict(
                            attn_resolutions=[],
                            ch=128,
                            ch_mult=[
                                1,
                                2,
                                4,
                                4,
                            ],
                            double_z=True,
                            dropout=0.0,
                            in_channels=3,
                            num_res_blocks=2,
                            out_ch=3,
                            resolution=256,
                            z_channels=4),
                        embed_dim=4,
                        lossconfig=dict(target='torch.nn.Identity'),
                        monitor='val/rec_loss'),
                    target='ldm.models.autoencoder.AutoencoderKL'),
                first_stage_key='jpg',
                image_size=64,
                linear_end=0.012,
                linear_start=0.00085,
                log_every_t=200,
                monitor='val/loss_simple_ema',
                num_timesteps_cond=1,
                scale_factor=0.18215,
                scheduler_config=dict(
                    params=dict(
                        cycle_lengths=[
                            10000000000000,
                        ],
                        f_max=[
                            1.0,
                        ],
                        f_min=[
                            1.0,
                        ],
                        f_start=[
                            1e-06,
                        ],
                        warm_up_steps=[
                            10000,
                        ]),
                    target='ldm.lr_scheduler.LambdaLinearScheduler'),
                timesteps=1000,
                unet_config=dict(
                    params=dict(
                        attention_resolutions=[
                            4,
                            2,
                            1,
                        ],
                        channel_mult=[
                            1,
                            2,
                            4,
                            4,
                        ],
                        context_dim=768,
                        image_size=32,
                        in_channels=4,
                        legacy=False,
                        model_channels=320,
                        num_heads=8,
                        num_res_blocks=2,
                        out_channels=4,
                        transformer_depth=1,
                        use_checkpoint=True,
                        use_spatial_transformer=True),
                    target='ldm.modules.diffusionmodules.openaimodel.UNetModel'
                ),
                use_ema=False),
            target='ldm.models.diffusion.ddpm.LatentDiffusion'),
        pad_shape=512,
        type='VPD',
        unet_cfg=dict(use_attn=False)),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            127.5,
            127.5,
            127.5,
        ],
        pad_val=0,
        seg_pad_val=0,
        size=(
            480,
            480,
        ),
        std=[
            127.5,
            127.5,
            127.5,
        ],
        type='SegDataPreProcessor'),
    decode_head=dict(
        fmap_border=(
            1,
            1,
        ),
        in_channels=[
            320,
            640,
            1280,
            1280,
        ],
        max_depth=10,
        type='VPDDepthHead'),
    test_cfg=dict(
        crop_size=(
            480,
            480,
        ), mode='slide_flip', stride=(
            160,
            160,
        )),
    type='DepthEstimator')
optim_wrapper = dict(
    clip_grad=None,
    constructor='ForceDefaultOptimWrapperConstructor',
    optimizer=dict(lr=0.001, type='AdamW', weight_decay=0.1),
    paramwise_cfg=dict(
        bias_decay_mult=0,
        custom_keys=dict({
            'backbone.encoder_vq': dict(lr_mult=0),
            'backbone.unet': dict(lr_mult=0.01)
        }),
        force_default_settings=True),
    type='OptimWrapper')
optimizer = dict(lr=0.001, type='AdamW', weight_decay=0.1)
param_scheduler = [
    dict(
        begin=0, by_epoch=False, end=12000, start_factor=0.03,
        type='LinearLR'),
    dict(
        begin=12000,
        by_epoch=False,
        end=24000,
        eta_min_ratio=0.03,
        power=0.9,
        type='PolyLRRatio'),
    dict(begin=24000, by_epoch=False, end=25000, factor=1, type='ConstantLR'),
]
resume = False
stable_diffusion_cfg = dict(
    base_learning_rate=0.0001,
    checkpoint=
    'https://download.openmmlab.com/mmsegmentation/v0.5/vpd/stable_diffusion_v1-5_pretrain_third_party.pth',
    params=dict(
        channels=4,
        cond_stage_config=dict(
            target='ldm.modules.encoders.modules.AbstractEncoder'),
        cond_stage_key='txt',
        cond_stage_trainable=False,
        conditioning_key='crossattn',
        first_stage_config=dict(
            params=dict(
                ddconfig=dict(
                    attn_resolutions=[],
                    ch=128,
                    ch_mult=[
                        1,
                        2,
                        4,
                        4,
                    ],
                    double_z=True,
                    dropout=0.0,
                    in_channels=3,
                    num_res_blocks=2,
                    out_ch=3,
                    resolution=256,
                    z_channels=4),
                embed_dim=4,
                lossconfig=dict(target='torch.nn.Identity'),
                monitor='val/rec_loss'),
            target='ldm.models.autoencoder.AutoencoderKL'),
        first_stage_key='jpg',
        image_size=64,
        linear_end=0.012,
        linear_start=0.00085,
        log_every_t=200,
        monitor='val/loss_simple_ema',
        num_timesteps_cond=1,
        scale_factor=0.18215,
        scheduler_config=dict(
            params=dict(
                cycle_lengths=[
                    10000000000000,
                ],
                f_max=[
                    1.0,
                ],
                f_min=[
                    1.0,
                ],
                f_start=[
                    1e-06,
                ],
                warm_up_steps=[
                    10000,
                ]),
            target='ldm.lr_scheduler.LambdaLinearScheduler'),
        timesteps=1000,
        unet_config=dict(
            params=dict(
                attention_resolutions=[
                    4,
                    2,
                    1,
                ],
                channel_mult=[
                    1,
                    2,
                    4,
                    4,
                ],
                context_dim=768,
                image_size=32,
                in_channels=4,
                legacy=False,
                model_channels=320,
                num_heads=8,
                num_res_blocks=2,
                out_channels=4,
                transformer_depth=1,
                use_checkpoint=True,
                use_spatial_transformer=True),
            target='ldm.modules.diffusionmodules.openaimodel.UNetModel'),
        use_ema=False),
    target='ldm.models.diffusion.ddpm.LatentDiffusion')
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
                2000,
                480,
            ), type='Resize'),
            dict(depth_rescale_factor=0.001, type='LoadDepthAnnotation'),
            dict(
                meta_keys=(
                    'img_path',
                    'depth_map_path',
                    'ori_shape',
                    'img_shape',
                    'pad_shape',
                    'scale_factor',
                    'flip',
                    'flip_direction',
                    'category_id',
                ),
                type='PackSegInputs'),
        ],
        test_mode=True,
        type=dataset_type),
    num_workers=0,
    persistent_workers=False,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    crop_type='nyu_crop',
    max_depth_eval=10.0,
    min_depth_eval=0.001,
    type='DepthMetric')
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(keep_ratio=True, scale=(
        2000,
        480,
    ), type='Resize'),
    dict(depth_rescale_factor=0.001, type='LoadDepthAnnotation'),
    dict(
        meta_keys=(
            'img_path',
            'depth_map_path',
            'ori_shape',
            'img_shape',
            'pad_shape',
            'scale_factor',
            'flip',
            'flip_direction',
            'category_id',
        ),
        type='PackSegInputs'),
]
train_cfg = dict(max_iters=25000, type='IterBasedTrainLoop', val_interval=1000)
train_dataloader = dict(
    batch_size=8,
    dataset=dict(
        data_prefix=dict(
            depth_map_path='annotations/train', img_path='images/train'),
        data_root='data/nyu',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(depth_rescale_factor=0.001, type='LoadDepthAnnotation'),
            dict(prob=0.25, type='RandomDepthMix'),
            dict(prob=0.5, type='RandomFlip'),
            dict(crop_size=(
                480,
                480,
            ), type='RandomCrop'),
            dict(
                transforms=[
                    dict(type='RandomBrightnessContrast'),
                    dict(type='RandomGamma'),
                    dict(type='HueSaturationValue'),
                ],
                type='Albu'),
            dict(
                meta_keys=(
                    'img_path',
                    'depth_map_path',
                    'ori_shape',
                    'img_shape',
                    'pad_shape',
                    'scale_factor',
                    'flip',
                    'flip_direction',
                    'category_id',
                ),
                type='PackSegInputs'),
        ],
        type='NYUDataset'),
    num_workers=8,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='InfiniteSampler'))
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(depth_rescale_factor=0.001, type='LoadDepthAnnotation'),
    dict(prob=0.25, type='RandomDepthMix'),
    dict(prob=0.5, type='RandomFlip'),
    dict(crop_size=(
        480,
        480,
    ), type='RandomCrop'),
    dict(
        transforms=[
            dict(type='RandomBrightnessContrast'),
            dict(type='RandomGamma'),
            dict(type='HueSaturationValue'),
        ],
        type='Albu'),
    dict(
        meta_keys=(
            'img_path',
            'depth_map_path',
            'ori_shape',
            'img_shape',
            'pad_shape',
            'scale_factor',
            'flip',
            'flip_direction',
            'category_id',
        ),
        type='PackSegInputs'),
]
tta_model = dict(type='SegTTAModel')
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_prefix=dict(
            depth_map_path='annotations/test', img_path='images/test'),
        data_root='data/nyu',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                2000,
                480,
            ), type='Resize'),
            dict(depth_rescale_factor=0.001, type='LoadDepthAnnotation'),
            dict(
                meta_keys=(
                    'img_path',
                    'depth_map_path',
                    'ori_shape',
                    'img_shape',
                    'pad_shape',
                    'scale_factor',
                    'flip',
                    'flip_direction',
                    'category_id',
                ),
                type='PackSegInputs'),
        ],
        test_mode=True,
        type='NYUDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    crop_type='nyu_crop',
    max_depth_eval=10.0,
    min_depth_eval=0.001,
    type='DepthMetric')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='SegLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])