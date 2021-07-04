import torch
import torch.nn as nn
import torch.nn.functional as F
from models.initializer import initialize_model
from algorithms.ERM import ERM
from algorithms.single_model_algorithm import SingleModelAlgorithm
from wilds.common.utils import split_into_groups
from configs.supported import process_outputs_functions, losses
import copy
from utils import load

class NoisyStudent(SingleModelAlgorithm):
    """
    Noisy Student.
    This algorithm was originally proposed as a semi-supervised learning algorithm.

    One run of this codebase gives us one iteration (load a teacher, train student). To run another iteration,
    re-run the previous command, pointing config.teacher_model_path to the trained student weights.

    Based on the original paper, loss is of the form
        \ell_s + \ell_u
    where 
        \ell_s = cross-entropy with true labels; student predicts with noise
        \ell_u = cross-entropy with pseudolabel generated without noise; student predicts with noise

    The student is noised using:
        - Input images are augmented using RandAugment
        - Single dropout layer before final classifier (fc) layer
        - TODO: stochastic depth with linearly decaying survival probability from last to first

    This code assumes that the teacher model is the same class as the student model (e.g. both densenet121s)

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
        assert config.soft_pseudolabels
        # load teacher model
        teacher_model = initialize_model(config, d_out).to(config.device)
        load(teacher_model, config.teacher_model_path, device=config.device)
        # initialize student model with dropout before last layer
        featurizer, classifier = initialize_model(config, d_out=d_out, is_featurizer=True) # note: pretrained on imagenet
        student_model = torch.nn.Sequential()
        setattr(student_model, 'features', featurizer)
        setattr(student_model, 'student_dropout', nn.Dropout(p=config.dropout_rate))
        setattr(student_model, 'classifier', classifier)
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
        
        if self.soft and d_out > 2:
            # Change loss to be the soft cross entropy 
            self.loss = losses[]

        if config.process_outputs_function is not None: 
            self.process_outputs_function = process_outputs_functions[config.process_outputs_function]
        # additional logging
        # set model components

    def state_dict(self):
        """
        Override the information that gets returned for saving in save_model
        Removes:
            - teacher state
            - student dropout layer
        """
        state = super().state_dict()
        state = { k:v for k,v in state.items() if 'teacher' not in k and 'student_dropout' not in k } # remove teacher & dropout info
        return state
        
    def process_batch(self, labeled_batch, unlabeled_batch=None):
        # TODO: for now, teacher takes in the same inputs as the student (same augs)
        # ideally, the data loader would yield: laebled, strongly augmented (student), normal unlabeled (teacher)
        # Labeled examples
        x, y_true, metadata = labeled_batch
        x = x.to(self.device)
        y_true = y_true.to(self.device)
        g = self.grouper.metadata_to_group(metadata).to(self.device)
        student_outputs = self.model(x)
        with torch.no_grad(): 
            teacher_outputs = self.teacher(x)
        results = {
            'g': g,
            'y_true': y_true,
            'y_pred': student_outputs,
            'metadata': metadata,
            'y_teacher': teacher_outputs,
        }
        # Unlabeled examples
        if unlabeled_batch is not None:
            x, metadata = unlabeled_batch
            x = x.to(self.device)
            g = self.grouper.metadata_to_group(metadata).to(self.device)
            with torch.no_grad(): 
                teacher_outputs = self.teacher(x)
            student_outputs = self.model(x)
            results['unlabeled_metadata'] = metadata
            results['unlabeled_y_pred'] = student_outputs
            results['unlabeled_y_teacher'] = teacher_outputs 
            results['unlabeled_g'] = g
        return results

    def objective(self, results):
        # Labeled loss
        import pdb
        pdb.set_trace()

        if self.hard_pseudolabels: teacher_outputs = self.process_outputs_function(teacher_outputs)

        labeled_loss = self.loss.compute(results['y_pred'], results['y_true'], return_dict=False)
        # Pseudolabeled loss
        if 'unlabeled_y_pred' in results: 
            unlabeled_loss = self.loss.compute(results['unlabeled_y_pred'], results['unlabeled_y_pseudo'], return_dict=False)
        else: unlabeled_loss = 0
        return labeled_loss + unlabeled_loss 
