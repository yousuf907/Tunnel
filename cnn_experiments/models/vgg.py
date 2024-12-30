"""VGG11/13/16/19 in Pytorch."""
import torch
import torch.nn as nn

## 32x32 resolutions ## without max-pool in the first 2 stages
#cfg = {
#    "VGG13": [64, 64, 128, 128, 256, 256, "M", 512, 512, "M", 512, 512, "M"],
#    "VGG19": [64, 64, 128, 128, 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 
#              512, 512, 512, 512, 'M']
#}

## 224x224 resolutions ## max-pool in all 5 stages
cfg = {
    "VGG13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG19": [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 
              512, 512, 512, 512, 'M']
}

class VGG(nn.Module):
    def __init__(self, vgg_name, class_num=100):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, class_num)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        nn.init.xavier_uniform_(self.classifier.weight)

    def forward(self, x):
        out = self.features(x)
        #print("shape of output before avg pool:", out.shape)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, x, kernel_size=3, padding=1)
                nn.init.kaiming_normal_(conv2d.weight, mode="fan_out", nonlinearity="relu")
                batchnorm = nn.BatchNorm2d(x)
                nn.init.constant_(batchnorm.weight, 1)
                nn.init.constant_(batchnorm.bias, 0)

                layers += [
                    conv2d,
                    batchnorm,
                    nn.ReLU(inplace=True),
                ]

                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)



if __name__ == "__main__":
    from types import SimpleNamespace
    
    #model = VGG("VGG13", class_num=100)
    model = VGG("VGG19", class_num=100)
    #print(model)

    #x = torch.randn((10, 3, 32, 32), dtype=torch.float32)
    #x = torch.randn((10, 3, 224, 224), dtype=torch.float32)
    #y = model(x)
    #print('Shape of y:', y.shape)

    n_parameters = sum(p.numel() for p in model.parameters())
    print('\nNumber of Params (in Millions):', n_parameters / 1e6)

    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())