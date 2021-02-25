

model_defaults = {
    'bert-base-uncased': {
        'optimizer': 'AdamW',
        'max_grad_norm': 1.0,
        'scheduler': 'linear_schedule_with_warmup',
    },
    'densenet121': {
        'model_kwargs':{
            'pretrained':True,
        },
        'target_resolution': (224, 224),
    },
    'wideresnet50': {
        'model_kwargs':{
            'pretrained':True,
        },
        'target_resolution': (224, 224),
    },
    'resnet50': {
        'model_kwargs':{
            'pretrained':True,
        },
        'target_resolution': (224, 224),
    },
    'gin-virtual': {},
    'resnet18_ms': {
        'target_resolution': (224, 224),
    },
    'efficientnet-b0': {
        'target_resolution': (224, 224),
    },
    'efficientnet-b1': {
        'target_resolution': (240, 240),
    },
    'efficientnet-b2': {
        'target_resolution': (260, 260),
    },
    'efficientnet-b3': {
        'target_resolution': (300, 300),
    },
    'efficientnet-b4': {
        'target_resolution': (380, 380),
    },
    'logistic_regression': {},
}
