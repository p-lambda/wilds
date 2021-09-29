from wilds.common.metrics.all_metrics import (
    Accuracy,
    MultiTaskAccuracy,
    MSE,
    multiclass_logits_to_pred,
    binary_logits_to_pred,
    pseudolabel_binary_logits,
    pseudolabel_multiclass_logits,
    pseudolabel_detection,
    MultiTaskAveragePrecision
)

algo_log_metrics = {
    'accuracy': Accuracy(prediction_fn=multiclass_logits_to_pred),
    'mse': MSE(),
    'multitask_accuracy': MultiTaskAccuracy(prediction_fn=multiclass_logits_to_pred),
    'multitask_binary_accuracy': MultiTaskAccuracy(prediction_fn=binary_logits_to_pred),
    'multitask_avgprec': MultiTaskAveragePrecision(prediction_fn=None),
    None: None,
}

process_outputs_functions = {
    'binary_logits_to_pred': binary_logits_to_pred,
    'multiclass_logits_to_pred': multiclass_logits_to_pred,
    None: None,
}

process_pseudolabels_functions = {
    'pseudolabel_binary_logits': pseudolabel_binary_logits,
    'pseudolabel_multiclass_logits': pseudolabel_multiclass_logits,
    'pseudolabel_detection': pseudolabel_detection,
}

# see initialize_*() functions for correspondence=
# See algorithms/initializer.py
algorithms = ['ERM', 'groupDRO', 'deepCORAL', 'IRM', 'DANN', 'FixMatch', 'PseudoLabel', 'NoisyStudent']

# See transforms.py
transforms = ['bert', 'image_base', 'image_resize', 'image_resize_and_center_crop', 'poverty',  'rxrx1']
additional_transforms = ['randaugment']

# See models/initializer.py
models = ['resnet18_ms', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'wideresnet50',
         'densenet121', 'bert-base-uncased', 'distilbert-base-uncased',
         'gin-virtual', 'logistic_regression', 'code-gpt-py',
         'fasterrcnn', 'unet-seq']

# See optimizer.py
optimizers = ['SGD', 'Adam', 'AdamW']

# See scheduler.py
schedulers = ['linear_schedule_with_warmup', 'cosine_schedule_with_warmup', 'ReduceLROnPlateau', 'StepLR', 'FixMatchLR', 'MultiStepLR']

# See losses.py
losses = ['cross_entropy', 'lm_cross_entropy', 'MSE', 'multitask_bce', 'fasterrcnn_criterion', 'cross_entropy_logits']
