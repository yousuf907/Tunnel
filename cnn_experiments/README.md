# Code for Training and Linear Probing CNN

Python code to reproduce CNN results presented in the paper What Variables Affect Out-Of-Distribution Generalization in Pretrained Models?

To use the code, simply run the following:

- To train CNNs on ID datasets with augmentations: `pretrain_aug.sh`

- To train CNNs on ID datasets without augmentations: `pretrain_no_aug.sh`

- For linear probing VGGs on ID and OOD datasets: `train_lp_vgg.sh`

- For linear probing ResNets on ID and OOD datasets: `train_lp_resnet.sh`

To get results for other configures, change the relevant arguments.
