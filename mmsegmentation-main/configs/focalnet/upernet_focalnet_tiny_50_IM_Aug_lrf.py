_base_ = [
    '../_base_/models/upernet_focalnet.py', '../_base_/datasets/IM_Aug.py',
    '../_base_/default_runtime.py'
]
model = dict(
    backbone=dict(
        type='FocalNet',
        embed_dim=96,
        depths=[2, 2, 6, 2],
        drop_path_rate=0.3,
        patch_norm=True,
        use_checkpoint=False,
        focal_windows=[9, 9, 9, 9],
        focal_levels=[3, 3, 3, 3],
    ),
    decode_head=dict(
        in_channels=[96, 192, 384, 768],
        num_classes=2
    ),
    auxiliary_head=dict(
        in_channels=384,
        num_classes=2
    ))

# optimizer
embed_multi = dict(lr_mult=1.0, decay_mult=0.0)
optimizer = dict(
    type='AdamW', lr=0.00006, weight_decay=0.01, eps=1e-8, betas=(0.9, 0.999))
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=optimizer,
    clip_grad=dict(max_norm=0.01, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1, decay_mult=1.0),
            'query_embed': embed_multi,
            'query_feat': embed_multi,
            'level_embed': embed_multi,
        },
        norm_decay_mult=0.0))

# learning policy
param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=0,
        power=1.0,
        begin=0,
        end=50,
        by_epoch=True)
]

# training schedule for 160k
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=50, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=True),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook', by_epoch=True, interval=50,
        save_best='mIoU'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook',
                       draw=True,
                       interval=1))

auto_scale_lr = dict(enable=False, base_batch_size=16)
