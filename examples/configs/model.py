model_defaults = {
    'bert-base-uncased': {
        'optimizer': 'AdamW',
        'max_grad_norm': 1.0,
        'scheduler': 'linear_schedule_with_warmup',
    },
    'distilbert-base-uncased': {
        'optimizer': 'AdamW',
        'max_grad_norm': 1.0,
        'scheduler': 'linear_schedule_with_warmup',
    },
    'code-gpt-py': {
        'optimizer': 'AdamW',
        'max_grad_norm': 1.0,
        'scheduler': 'linear_schedule_with_warmup',
    },
    'densenet121': {
        'model_kwargs': {
            'pretrained':True,
        },
        'target_resolution': (224, 224),
    },
    'wideresnet50': {
        'model_kwargs': {
            'pretrained':True,
        },
        'target_resolution': (224, 224),
    },
    'resnet50': {
        'model_kwargs': {
            'pretrained':True,
        },
        'target_resolution': (224, 224),
    },
    'gin-virtual': {},
    'resnet18_ms': {
        'target_resolution': (224, 224),
    },
    'logistic_regression': {},
    'detr': {
        'max_grad_norm': 0.1,
        'model_kwargs': {
            # Backbone. Always uses sine position embedding.
            'train_backbone': True,
            'backbone': 'resnet50',
            'dilation': False,
            # Transformer
            'enc_layers': 6,
            'dec_layers': 6,
            'dim_feedforward': 2048,
            'hidden_dim': 256,
            'dropout': 0.1,
            'nheads': 8,
            'pre_norm': False,
        },
        'loss_kwargs': {
            # Matcher
            'set_cost_class': 1,
            'set_cost_bbox': 5,
            'set_cost_giou': 2,
            # Loss
            'mask_loss_coef': 1,
            'dice_loss_coef': 1,
            'bbox_loss_coef': 5,
            'giou_loss_coef': 2,
            # 'eos_coef': 0.1,
            'eos_coef': 0.5,
        }
    },
    'fasterrcnn': {
        'model_kwargs': {
            # Backbone. Always uses sine position embedding.
            'pretrained': True,
        }
    }
}
