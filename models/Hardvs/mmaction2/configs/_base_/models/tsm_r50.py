# model settings
model = dict(
    type='Recognizer2D',
    backbone=dict(
        type='ResNetTSM',
        # pretrained='torchvision://resnet50',
        pretrained=None,
        depth=50,
        norm_eval=False,
        shift_div=8,
        num_segments=16
        # num_segments=32
        ),
    cls_head=dict(
        type='TSMHead',
        # num_classes=400,
        num_classes=200,
        # num_classes=50,
        # num_classes=300,
        in_channels=2048,
        spatial_type='avg',
        consensus=dict(type='AvgConsensus', dim=1),
        dropout_ratio=0.5,
        init_std=0.001,
        is_shift=True,
        num_segments=16
        # num_segments=32
        ),
    # model training and testing settings
    train_cfg=None,
    test_cfg=dict(average_clips='prob'))
