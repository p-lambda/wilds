import torch
from algorithms.single_model_algorithm import SingleModelAlgorithm
from models.initializer import initialize_model

class ERM(SingleModelAlgorithm):
    def __init__(self, config, d_out, grouper, loss,
            metric, n_train_steps):
        model = initialize_model(config, d_out)
        # initialize module
        super().__init__(
            config=config,
            model=model,
            grouper=grouper,
            loss=loss,
            metric=metric,
            n_train_steps=n_train_steps,
        )
        self.use_unlabeled_y = config.use_unlabeled_y # Expect x,y,m from unlabeled loaders and train on the unlabeled y

    def process_batch(self, batch, unlabeled_batch=None):
        """
        A helper function for update() and evaluate() that processes the batch.
        ERM defines its own process_batch to handle if self.use_unlabeled_y is true.
        Args:
            - batch (tuple of Tensors): a batch of data yielded by data loaders
            - unlabeled_batch (tuple of Tensors or None): a batch of data yielded by unlabeled data loader
        Output:
            - results (dictionary): information about the batch
                - y_true (Tensor)
                - g (Tensor)
                - metadata (Tensor)
                - output (Tensor)
        """
        x, y_true, metadata = batch
        x = move_to(x, self.device)
        y_true = move_to(y_true, self.device)
        g = move_to(self.grouper.metadata_to_group(metadata), self.device)

        outputs = self.get_model_output(x, y_true)

        results = {
            'g': g,
            'y_true': y_true,
            'y_pred': outputs,
            'metadata': metadata,
        }
        if unlabeled_batch is not None:
            if self.use_unlabeled_y: # expect loaders to return x,y,m
                x, y, metadata = unlabeled_batch
                y = move_to(y, self.device)
            else:
                x, metadata = unlabeled_batch    
            x = move_to(x, self.device)
            results['unlabeled_metadata'] = metadata
            if self.use_unlabeled_y:
                results['unlabeled_y_pred'] = self.get_model_output(x, y)
                results['unlabeled_y_true'] = y
            results['unlabeled_g'] = self.grouper.metadata_to_group(metadata).to(self.device)
        return results

    def objective(self, results):
        labeled_loss = self.loss.compute(results['y_pred'], results['y_true'], return_dict=False)
        if not config.use_unlabeled_y:
            return labeled_loss
        else:
            return labeled_loss + self.loss.compute(results['unlabeled_y_pred'], results['unlabeled_y_true'], return_dict=False)
