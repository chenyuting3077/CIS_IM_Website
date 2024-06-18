_base_ = [
    '../_base_/models/upernet_vit-b16_ln_mln.py', '../_base_/datasets/IM.py',
    '../_base_/default_runtime.py'
]
crop_size = (224, 224)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    pretrained='pretrain/vit_base_patch16_224.pth',
    decode_head=dict(num_classes=2),
    auxiliary_head=dict(num_classes=2))

optim_wrapper = dict(

    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=1e-4, betas=(0.9, 0.999), weight_decay=0.05),
    # paramwise_cfg=dict(num_layers=12, layer_decay_rate=0.65),
    )

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=True, begin=0, end=10),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=10,
        end=500,
        by_epoch=True,
    )
]

# mixed precision
fp16 = dict(loss_scale='dynamic')

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


# By default, models are trained on 8 GPUs with 2 images per GPU
train_dataloader = dict(batch_size=2)
val_dataloader = dict(batch_size=1)
test_dataloader = val_dataloader
