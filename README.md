# CoupleVAE

This is the official Implementation for our paper:

CoupleVAE: a coupling variational autoencoders model for predicting perturbational single-cell RNA sequencing data

Yahao Wu, Songyan Liu, Limin Li

![image](https://github.com/muxiaran6688/XJTU/blob/main/img/CoupleVAE.PNG)

    
## Getting Started

To run the CoupleVAE you need following packages :
### `Requirements`

    python                                               3.6 
    anndata                                              0.7.4
    scanpy                                               1.6.0
    tensorflow                                           2.0.0
    numpy                                                1.20.3
    scipy                                                1.5.3
    pandas                                               1.1.3
    matplotlib                                           3.4.3
    seaborn                                              0.11.2
    
## Installation

install the development version via pip:
```bash
pip install git+https://github.com/LiminLi-XJTU/CoupleVAE.git
```

## Example

Datasets from our paper are available in 

```bash
cd code/
python train_couplevae.py

```
Then you can complete the training process and get the predicted dataYou can complete the training process and get the predicted data.
