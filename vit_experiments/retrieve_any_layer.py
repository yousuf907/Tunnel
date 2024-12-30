import torch
import torch.nn as nn
#from torchvision.models.resnet import resnet18
from utils_cl import *

def get_name_to_module(model):
    name_to_module = {}
    for m in model.named_modules():
        name_to_module[m[0]] = m[1]
    return name_to_module


def get_activation(all_outputs, name):
    def hook(model, input, output):
        all_outputs[name] = output.detach()

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
        output_vals = [self.outputs[output_layer_name] for output_layer_name in self.output_layer_names]
        if self.return_single:
            return output_vals[0]
        else:
            return output_vals


def test_resnet18():
    output_layer_names = ['layer1.0.bn1', 'layer4.0', 'fc']
    #in_tensor = torch.ones((2, 3, 224, 224))
    in_tensor = torch.ones((2, 3, 32, 32))

    #core_model = resnet18()
    core_model = resnet18(num_classes=50, affine=True)
    wrapper = ModelWrapper(core_model, output_layer_names)
    y1, y2, y3 = wrapper(in_tensor)
    #print(y1.shape)
    #print(y2.shape)
    #print(y3.shape)
    assert y1.shape[0] == 2
    assert y1.shape[2] == 32 #56
    assert y2.shape[2] == 4 #7
    assert y3.shape[1] == 50 #1000


if __name__ == "__main__":
    #test_resnet18()
    print("model defined")

'''
### ResNet18 Layers ###
[
"conv1",
"layer1.0.conv1",
"layer1.0.conv2",
"layer1.1.conv1",
"layer1.1.conv2",
"layer2.0.conv1",
"layer2.0.conv2",
"layer2.1.conv1",
"layer2.1.conv2",
"layer3.0.conv1",
"layer3.0.conv2",
"layer3.1.conv1", # 12 > extractor 1
"layer3.1.conv2",
"layer4.0.conv1", # 14 > extractor 2
"layer4.0.conv2",
"layer4.1.conv1",
"layer4.1.conv2",
"fc"
]
'''