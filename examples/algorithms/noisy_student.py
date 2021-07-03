import torch
import torch.nn.functional as F
from models.initializer import initialize_model
from algorithms.ERM import ERM
from algorithms.single_model_algorithm import SingleModelAlgorithm
from wilds.common.utils import split_into_groups
from configs.supported import process_outputs_functions
import copy
from utils import load

class NoisyStudent(SingleModelAlgorithm):
    """
    Noisy Student.
    This algorithm was originally proposed as a semi-supervised learning algorithm. 

    Assumes that the teacher model is the same class as the student model. 

    Original paper:
        @inproceedings{xie2020self,
            title={Self-training with noisy student improves imagenet classification},
            author={Xie, Qizhe and Luong, Minh-Thang and Hovy, Eduard and Le, Quoc V},
            booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
            pages={10687--10698},
            year={2020}
            }
    """
    def __init__(self, config, d_out, grouper, loss, metric, n_train_steps):
        # check config
        assert config.teacher_model_path is not None
        # load teacher model
        teacher_model = initialize_model(config, d_out).to(config.device)
        load(teacher_model, config.teacher_model_path, device=config.device)
        # initialize student model
        student_model = initialize_model(config, d_out=d_out) # note: pretrained on imagenet
        student_model = student_model.to(config.device)
        # initialize module
        super().__init__(
            config=config,
            model=student_model,
            grouper=grouper,
            loss=loss,
            metric=metric,
            n_train_steps=n_train_steps,
        )
        self.teacher = teacher_model
        # algorithm hyperparameters
        self.hard_pseudolabels = True
        if config.process_outputs_function is not None: 
            self.process_outputs_function = process_outputs_functions[config.process_outputs_function]
        # additional logging
        # set model components

    def process_batch(self, labeled_batch, unlabeled_batch=None):
        # TODO: for now, teacher takes in the same inputs as the student (same augs)
        # ideally, the data loader would yield: laebled, strongly augmented (student), normal unlabeled (teacher)
        # Student learns from labeled examples
        x, y_true, metadata = labeled_batch
        x = x.to(self.device)
        y_true = y_true.to(self.device)
        g = self.grouper.metadata_to_group(metadata).to(self.device)
        outputs = self.model(x)
        # package the results
        results = {
            'g': g,
            'y_true': y_true,
            'y_pred': outputs,
            'metadata': metadata
        }
        # Student learns from pseudolabeled examples
        if unlabeled_batch is not None:
            x, metadata = unlabeled_batch
            x = x.to(self.device)
            g = self.grouper.metadata_to_group(metadata).to(self.device)
            with torch.no_grad(): 
                teacher_outputs = self.teacher(x)
                if self.hard_pseudolabels: teacher_outputs = self.process_outputs_function(teacher_outputs)
            student_outputs = self.model(x)
            results['unlabeled_metadata'] = metadata
            results['unlabeled_y_pseudo'] = teacher_outputs 
            results['unlabeled_y_pred'] = student_outputs
            results['unlabeled_g'] = g
        return results

    def objective(self, results):
        # TODO: check the scaling that the original paper does based on labeled v. unlabeled batch size
        # Labeled loss
        labeled_loss = self.loss.compute(results['y_pred'], results['y_true'], return_dict=False)
        # Pseudolabeled loss
        if 'unlabeled_y_pred' in results: 
            unlabeled_loss = self.loss.compute(results['unlabeled_y_pred'], results['unlabeled_y_pseudo'], return_dict=False)
        else: unlabeled_loss = 0
        return labeled_loss + unlabeled_loss 
