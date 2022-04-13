config = dict(
    model=dict(
        type='RSSGL',
        params=dict(
            in_channels=200,
            num_classes=16,
            block_channels=(96, 128, 192, 256),
            inner_dim=128,
            reduction_ratio=1.0,
        )
    ),
    data=dict(
        train=dict(
            type='NewIndianPinesLoader',
            params=dict(
                training=True,
                num_workers=4,
                image_mat_path='./IndianPines/Indian_pines_corrected.mat',
                gt_mat_path='./IndianPines/Indian_pines_gt.mat',
                sample_percent=0.05,
                batch_size=10
            )
        ),
        test=dict(
            type='NewIndianPinesLoader',
            params=dict(
                training=False,
                num_workers=4,
                image_mat_path='./IndianPines/Indian_pines_corrected.mat',
                gt_mat_path='./IndianPines/Indian_pines_gt.mat',
                sample_percent=0.05,
                batch_size=10
            )
        )
    ),
    optimizer=dict(
        type='sgd',
        params=dict(
            momentum=0.9,
            weight_decay=0.001
        )
    ),
    learning_rate=dict(
        type='poly',
        params=dict(
            base_lr=0.005,
            power=0.8,
            max_iters=1000),
    ),
    train=dict(
        forward_times=1,
        num_iters=1000,
        eval_per_epoch=True,
        summary_grads=False,
        summary_weights=False,
        eval_after_train=True,
        resume_from_last=False,
        early_stopping = True,
        early_epoch = 0,
        early_num = 15,
        PATH = "./Optimal_Indian_Pines.pt",
        test_oa = [0.]
    ),
    test=dict(
        draw=dict(
            image_size=(145, 145),
            palette=[
                [0, 0, 0],
                [255, 0, 0],
                [0, 255, 0],
                [0, 0, 255],
                [255, 255, 0],
                [0, 255, 255],
                [255, 0, 255],
                [192, 192, 192],
                [128, 128, 128],
                [128, 0, 0],
                [128, 128, 0],
                [0, 128, 0],
                [128, 0, 128],
                [0, 128, 128],
                [0, 0, 128],
                [255, 165, 0],
                [255, 215, 0]
            ]
        )
    ),
)
