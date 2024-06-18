_base_ = [
    '../_base_/models/upernet_focalnet.py', '../_base_/datasets/IMLargeShuffle.py',
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
        focal_levels=[2, 2, 2, 2],
    ),
    decode_head=dict(
        in_channels=[96, 192, 384, 768],
        num_classes=2
    ),
    auxiliary_head=dict(
        in_channels=384,
        num_classes=2
    ))

# AdamW optimizer, no weight decay for position embedding & layer norm in backbone
optimizer = dict(
    type='AdamW', lr=0.0001, weight_decay=0.05, eps=1e-8, betas=(0.9, 0.999))

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=optimizer,
    clip_grad=dict(max_norm=0.01, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)},
        norm_decay_mult=0.0)
    )



# training schedule for 160k
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=500, val_interval=50)
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

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=16)
