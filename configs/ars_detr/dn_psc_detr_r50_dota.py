# -*- coding: utf-8 -*-

angle_version = 'le90'
_base_ = [
    '../_base_/datasets/dotav1.py', '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
] 
model = dict(
    type='DNARSDETR',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='ChannelMapper',
        in_channels=[512, 1024, 2048],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=4),
    bbox_head=dict(
        type='DNARSDeformableDETRHead',
        num_query=300,
        # num_classes=15,
        num_classes=1,
        in_channels=2048,
        sync_cls_avg_factor=True,
        with_box_refine=True,
        as_two_stage=True,
        
        dn_cfg=dict(
            type = 'DnQueryGenerator',
            noise_scale=dict(label=0.5, box=0.4, angle=0.02),
            group_cfg=dict(dynamic=True, num_groups=None, num_dn_queries=300)
        ),
        
        # angle_coder=dict(
        #     type='CSLCoder',
        #     angle_version=angle_version,
        #     omega=1,
        #     window='gaussian',
        #     radius=6),
        angle_coder=dict(
            type='PSCCoder',
            angle_version=angle_version,
            dual_freq=True,
            num_step=3),
        transformer=dict(
            type='DNARSRotatedDeformableDetrTransformer',
            two_stage_num_proposals=300,
            encoder=dict(
                type='DetrTransformerEncoder',
                num_layers=6,
                transformerlayers=dict(
                    type='BaseTransformerLayer',
                    attn_cfgs=dict(
                        type='MultiScaleDeformableAttention', embed_dims=256),
                    feedforward_channels=1024,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'ffn', 'norm'))),
            decoder=dict(
                type='DNARSDeformableDetrTransformerDecoder',
                num_layers=6,
                return_intermediate=True,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        dict(
                            type='MultiScaleDeformableAttention',
                            embed_dims=256)
                    ],
                    feedforward_channels=1024,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')))),
        positional_encoding=dict(
            type='SinePositionalEncoding',
            num_feats=128,
            normalize=True,
            offset=-0.5),
        bbox_coder=dict(
            type='DeltaXYWHAOBBoxCoder',
            angle_range=angle_version,
            norm_factor=None,
            edge_swap=True,
            proj_xy=True,
            target_means=(.0, .0, .0, .0, .0),
            target_stds=(1, 1, 1, 1, 1)),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=2.0),
        loss_iou=dict(type='GIoULoss', loss_weight=5.0),
        reg_decoded_bbox=True,
        # loss_angle=dict(
        #     type='CrossEntropyLoss', use_sigmoid=True, loss_weight=2.0),
        loss_angle = dict(type='L1Loss', loss_weight=0.2),
    ),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='ARS_HungarianAssigner',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBoxL1Cost', weight=2.0, box_format='xywh'),
            iou_cost=dict(type='IoUCost', iou_mode='giou', weight=5.0),
            # angle_cost=dict(type='CrossEntropyLossCost', weight=2.0)
            angle_cost=dict(type='AngleL1Cost', weight=0.2)
        )),
    test_cfg=dict()
)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RResize', img_scale=(1024, 1024)),
    dict(
        type='RRandomFlip',
        flip_ratio=[0.25, 0.25, 0.25],
        direction=['horizontal', 'vertical', 'diagonal'],
        version=angle_version),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(pipeline=train_pipeline, version=angle_version),
    val=dict(version=angle_version),
    test=dict(version=angle_version))

optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=1e-4,
    weight_decay=0.00001,
    betas=(0.9, 0.999),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1),
            'sampling_offsets': dict(lr_mult=0.1),
            'reference_points': dict(lr_mult=0.1)
        }))
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
# lr_config = dict(policy='step', step=[32])
lr_config = dict(policy='step', step=[40])
runner = dict(type='EpochBasedRunner', max_epochs=100)
checkpoint_config = dict(interval=1)
evaluation = dict(interval=1,
                  save_best='auto',
                  metric='mAP')
find_unused_parameters = True

# GED 数据
# work_dir = 'work_dirs/dn_psc_new_refine/'
work_dir = 'work_dirs/test_projects/'
