import torch
import torch.nn as nn
import torch.nn.functional as F
#from torchvision.models.resnet import resnet18
from models.vgg import *

def get_name_to_module(model):
    name_to_module = {}
    for m in model.named_modules():
        name_to_module[m[0]] = m[1]
    return name_to_module

def get_activation(all_outputs, name):
    def hook(model, input, output):
        #all_outputs[name] = output.detach()
        #all_outputs[name] = torch.flatten(F.adaptive_avg_pool2d(output, 2).squeeze(), 1).detach()
        if output.shape[2] > 2:
            all_outputs[name] = torch.flatten(F.adaptive_avg_pool2d(output, 2).squeeze(), 1).detach()
        else:
            all_outputs[name] = torch.flatten(output.squeeze(), 1).detach()

    return hook


def add_hooks(model, outputs, output_layer_names):
    """
    :param model:
    :param outputs: Outputs from layers specified in `output_layer_names` will be stored in `output` variable
    :param output_layer_names:
    :return:
    """
    name_to_module = get_name_to_module(model)
    for output_layer_name in output_layer_names:
        name_to_module[output_layer_name].register_forward_hook(get_activation(outputs, output_layer_name))


class ModelWrapper(nn.Module):
    def __init__(self, model, output_layer_names, return_single=False):
        super(ModelWrapper, self).__init__()
        self.model = model
        self.output_layer_names = output_layer_names
        self.outputs = {}
        self.return_single = return_single
        add_hooks(self.model, self.outputs, self.output_layer_names)

    def forward(self, images):
        self.model(images)
        output_vals = {}
        #output_vals = [self.outputs[output_layer_name] for output_layer_name in self.output_layer_names]
        for name in self.output_layer_names:
            if int(name[9:]) < 10:
                output_vals[name.replace('.', '0')] = self.outputs[name]
            else:
                output_vals[name.replace('.', '')] = self.outputs[name]
            
        #if self.return_single:
        #    return output_vals[0]
        #else:
        #    return output_vals
        return output_vals

'''
def test_resnet18():
    output_layer_names = ['layer1.0.bn1', 'layer4.0', 'fc']
    in_tensor = torch.ones((2, 3, 224, 224))

    core_model = resnet18()
    wrapper = ModelWrapper(core_model, output_layer_names)
    y1, y2, y3 = wrapper(in_tensor)
    assert y1.shape[0] == 2
    assert y1.shape[2] == 56
    assert y2.shape[2] == 7
    assert y3.shape[1] == 1000
'''

def test_vgg13():
    #output_layer_names = ['features.0']
    output_layer_names = [
                        "features.0",
                        "features.3",
                        "features.6",
                        "features.9",
                        "features.12",
                        "features.15",
                        "features.19",
                        "features.22",
                        "features.26",
                        "features.29"
                    ]

    '''
    output_layer_names = [
                        "features.0",
                        "features.3",
                        "features.7",
                        "features.10",
                        "features.14",
                        "features.17",
                        "features.21",
                        "features.24",
                        "features.28",
                        "features.31"
                    ]
    '''

    #in_tensor = torch.ones((2, 3, 224, 224))
    in_tensor = torch.ones((2, 3, 32, 32))

    core_model = VGG("VGG13", class_num=10)
    wrapper = ModelWrapper(core_model, output_layer_names)
    #y1,y2,y3,y4,y5,y6,y7,y8,y9,y10 = wrapper(in_tensor)
    y1 = wrapper(in_tensor)
    print(len(y1))
    for k in y1:
        print(y1[k].shape)
    #assert y1.shape[0] == 2
    #assert y1.shape[2] == 56
    #assert y2.shape[2] == 7
    #assert y3.shape[1] == 1000


if __name__ == "__main__":
    #test_resnet18()
    test_vgg13()
