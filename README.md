# XJTU
# VAE for pertubational single cell data

This model is used to predict the gene expression of cells after a certain perturbation.

Dataset：COVID-19 dataset. After the dataset was processed, only macrophages were retained, which contained 3800 cells, 5598 genes and two pertubation condition(conotrol, moderate COVID-19).
3040 cells are used as training set and 380 cells are used as test set. The entire dataset is a gene expression matrix, with rows being cells and columns being genes.

Model:VAE

![image](https://github.com/muxiaran6688/XJTU/blob/main/img/CoupleVAE.PNG)

Target:We hope the model can accurately predict the gene expression of cells after Covid-19 infection.

To run the notebooks and scripts you need following packages :

tensorflow, scanpy, numpy, matplotlib, scipy, wget.


## Getting Started

```bash
cd code/
python train_VAE.py

```

Then you can run each notebook and get the results.

# **ViewFormer**

This is the PyTorch Implementation for our paper:

ViewFormer:Few-Shot Learning Baesd on Multiple Views

Jing Liu, Xi Wang, Limin Li

< img src="figs/frame.jpg">


## `Requirements`

    python                                               3.6 
    CUDA Version                                         10.0.130
    Pytorch                                              1.7.0
    torchvision                                          0.8.1
    timm                                                 0.4.9
    Pillow                                               8.1.0
    opencv-contrib-python                                4.5.2.52
 





## Multi-View Few-shot Learning 

For the Multi-View Few-shot Learning task, we are given a set of samples with multiple views. 
During training, only samples of a subset of support set are available. For testing, the model is tested
on samples of unseen (novel) class. We evaluate ViewFormer on three datasets, 
MVset, Caltech-20 and NUS-WIDE-OBJECT. 


### Data Statistics

< img src="figs/img.png">

To train the model, you can simply run
```angular2html
python train.py --model  MULT  --when 1 --num_epochs 10  --train_num 150 --val_num 100 --test_num 100   --name ViewFormer --lr 1.0e-04 --num_heads 64  --nlevels 6  --shot 5 --model_root ./model   --lamda 1 --beta 0.1  --view_encoding --shot_encoding
```
For testing, 
```angular2html
python test_proto.py --model MULT  --test_num 100   --name ViewFormer --num_heads 64  --nlevels 6 --model_root /home/datasets/User_Data/lj/paper_model --shot 1  --view_encoding --shot_encoding
python test_LR.py --model MULT  --test_num 100   --name ViewFormer --num_heads 64  --nlevels 6 --model_root /home/datasets/User_Data/lj/paper_model --shot 1  --view_encoding --shot_encoding
