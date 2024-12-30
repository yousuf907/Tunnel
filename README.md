The Tunnel Effect and OOD Generalization
========================================
Source Code for Our NeurIPS-2024 Paper "What Variables Affect Out-of-Distribution Generalization in Pretrained Models?"
Check out our paper: [arXiv link](https://arxiv.org/abs/2405.15018)

![Tunnel Effect](./tunnel_effect.png)

Embeddings produced by pre-trained deep neural networks (DNNs) are widely used; however, their efficacy for downstream tasks can vary widely. We study the factors influencing transferability and out-of-distribution (OOD) generalization of pre-trained DNN embeddings through the lens of the tunnel effect hypothesis, which is closely related to intermediate neural collapse. This hypothesis suggests that deeper DNN layers compress representations and hinder OOD generalization. We comprehensively investigate the impact of DNN architecture, training data, image resolution, and augmentations on transferability. We identify that training with high-resolution datasets containing many classes greatly reduces representation compression and improves transferability. Our results emphasize the danger of generalizing findings from toy datasets to broader contexts.


## Dependencies

The conda environment that we used for this project has been shared in the GitHub repository. 
The yml file `environment.yml` includes all the libraries. We have tested the code with the packages and versions specified in the yml file. Our ViT experiments require the timm library, `pip install timm`.
We recommend setting up a `conda` environment using the `environment.yml` file:
```
conda env create -f environment.yml
```
Our SHAP analysis requires the SHAP package. You can install it from [here](https://shap.readthedocs.io/en/latest/).


## Citation
If using this code, please cite our paper.
```
@article{harun2024variables,
  title={What Variables Affect Out-Of-Distribution Generalization in Pretrained Models?},
  author={Harun, Md Yousuf and Lee, Kyungbok and Gallardo, Jhair and Krishnan, Giri and Kanan, Christopher},
  journal={arXiv preprint arXiv:2405.15018},
  year={2024}
}
```

