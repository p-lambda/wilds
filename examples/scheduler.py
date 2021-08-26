from transformers import get_linear_schedule_with_warmup
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau, StepLR, CosineAnnealingLR

def initialize_scheduler(config, optimizer, n_train_steps):
    # construct schedulers
    if config.scheduler is None:
        return None
    elif config.scheduler == 'linear_schedule_with_warmup':
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_training_steps=n_train_steps,
            **config.scheduler_kwargs)
        step_every_batch = True
        use_metric = False
    elif config.scheduler == 'ReduceLROnPlateau':
        assert config.scheduler_metric_name, f'scheduler metric must be specified for {config.scheduler}'
        scheduler = ReduceLROnPlateau(
            optimizer,
            **config.scheduler_kwargs)
        step_every_batch = False
        use_metric = True
    elif config.scheduler == 'StepLR':
        scheduler = StepLR(optimizer, **config.scheduler_kwargs)
        step_every_batch = False
        use_metric = False
    elif config.scheduler == 'FixMatchLR':
        scheduler = LambdaLR(
            optimizer,
            lambda x: (1.0 + 10 * float(x) / n_train_steps) ** -0.75
        )
        step_every_batch = True
        use_metric = False
    elif config.scheduler == 'CosineLR':
        scheduler = CosineAnnealingLR(
            optimizer,
            n_train_steps,
            config.scheduler_kwargs['min_lr'] * config.lr
        )
        step_every_batch = True
        use_metric = False
    else:
        raise ValueError(f'Scheduler: {config.scheduler} not supported.')

    # add an step_every_batch field
    scheduler.step_every_batch = step_every_batch
    scheduler.use_metric = use_metric
    return scheduler

def step_scheduler(scheduler, metric=None):
    print(scheduler, isinstance(scheduler, ReduceLROnPlateau))
    if isinstance(scheduler, ReduceLROnPlateau):
        assert metric is not None
        scheduler.step(metric)
    else:
        scheduler.step()

class LinearScheduleWithWarmupAndThreshold():
    """
    Linear scheduler with warmup and threshold for non lr parameters. 
    Parameters is held at 0 until some T1, linearly increased until T2, and then held
    at some max value after T2.
    Designed to be called by step_scheduler() above and used within Algorithm class.
    Args:
        - num_warmup_steps: aka T1. for steps [0, T1) keep param = 0
        - threshold_step: aka T2. step over period [T1, T2) to reach param = max value 
        - max value: end value of the param
    """
    def __init__(self, max_value, num_warmup_steps=0, threshold_step=1, step_every_batch=False):
        self.max_value = max_value
        self.T1 = num_warmup_steps
        self.T2 = threshold_step
        assert (0 <= self.T1) and (self.T1 < self.T2)

        # internal tracker of which step we're on
        self.current_step = 0
        self.value = 0

        # required fields called in Algorithm when stepping schedulers
        self.step_every_batch = step_every_batch
        self.use_metric = False

    def step():
        print("Called step")
        if self.current_step < self.T1:
            self.value = 0
        elif self.current_step < self.T2:
            self.value += (self.current_step - self.T1) / (self.T2 - self.T1) * self.max_value
        else:
            self.value = self.max_value
        self.current_step += 1