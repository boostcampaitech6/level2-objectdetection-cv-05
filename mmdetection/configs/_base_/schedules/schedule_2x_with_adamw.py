# optimizer
optimizer = dict(constructor='LearningRateDecayOptimizerConstructor', type='AdamW', 
                 lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05,
                 paramwise_cfg={'decay_rate': 0.7,
                                'decay_type': 'layer_wise',
                                'num_layers': 12})

optimizer_config = dict(grad_clip=None)
# learning policy

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[16, 22])

runner = dict(type='EpochBasedRunner', max_epochs=24)
