from . import resnet
from . import densenet

def get_model(arch, **kwargs):
    model_dict = {}
    model_dict.update(resnet.__dict__)
    model_dict.update(densenet.__dict__)
    return model_dict[arch](**kwargs)
