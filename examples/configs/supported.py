# metrics
from wilds.common.metrics.all_metrics import Accuracy, MultiTaskAccuracy, MSE, multiclass_logits_to_pred, binary_logits_to_pred
from utils import remove_key

algo_log_metrics = {
    'accuracy': Accuracy(prediction_fn=multiclass_logits_to_pred),
    'mse': MSE(),
    'multitask_accuracy': MultiTaskAccuracy(prediction_fn=multiclass_logits_to_pred),
    'multitask_binary_accuracy': MultiTaskAccuracy(prediction_fn=binary_logits_to_pred),
    None: None,
}

process_outputs_functions = {
    'binary_logits_to_pred': binary_logits_to_pred,
    'multiclass_logits_to_pred': multiclass_logits_to_pred,
    'remove_detr_aux_outputs': remove_key('aux_outputs'),
    None: None,
}

# See models/initializer.py
models = ['resnet18_ms', 'resnet50', 'resnet34', 'wideresnet50',
         'densenet121', 'bert-base-uncased', 'distilbert-base-uncased',
         'gin-virtual', 'logistic_regression', 'code-gpt-py',
         'detr']

# See algorithms/initializer.py
algorithms = ['ERM', 'groupDRO', 'deepCORAL', 'IRM']

# See optimizer.py
optimizers = ['SGD', 'Adam', 'AdamW']

# See scheduler.py
schedulers = ['linear_schedule_with_warmup', 'ReduceLROnPlateau', 'StepLR']

# See transforms.py
transforms = ['bert', 'image_base', 'image_resize_and_center_crop', 'poverty_train']

# See losses.py
losses = ['cross_entropy', 'lm_cross_entropy', 'MSE', 'multitask_bce', 'detr_set_criterion']
