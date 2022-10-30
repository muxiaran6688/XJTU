# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 00:23:20 2022

@author: 70772
"""

import sys
import numpy as np
import scanpy as sc
import anndata
from scipy import stats
import os
import trvae

# data_name = 'covid'
# if data_name == "covid":
#     train_path = "./data/trvaetrain_covid_NK.h5ad"
#     # test_path = "./data/test_covid_NK.h5ad"
#     conditions = ["control", "severe COVID-19"]
#     source_condition = "control"
#     target_condition = "severe COVID-19"
#     labelencoder = {"control": 0, "severe COVID-19": 1}
#     cell_type_key = "celltype"
#     condition_key = "condition"


# else:
#     raise Exception("InValid data name")

# data = sc.read(train_path)
# # test_adata = sc.read(test_path)
# # train_adata = train_adata[train_adata.obs[condition_key].isin(conditions)]

# if data.shape[1] > 2000:
#     sc.pp.highly_variable_genes(data, n_top_genes=2000)
#     data = data[:, data.var['highly_variable']]
#     train_adata = data[data.obs['celltype']=="NK1"]
#     train_adata.obs['celltype'].replace("NK1","NK",inplace=True)
#     test_adata = data[data.obs['celltype']=="NK"]
# print(test_adata)
cell_type_list=["B","NK","CD14+Mono","FCGR3A+Mono","CD4T","CD8T","Dendritic"]
for specific_celltype in cell_type_list:
    data_name = 'pbmc'
    train_path = f"/home/datasets/psvae/train_{data_name}_{specific_celltype}.h5ad"
    test_path = f"/home/datasets/psvae/test_{data_name}_{specific_celltype}.h5ad"
    conditions = ["control", "stimulated"]
    source_condition = "control"
    target_condition = "stimulated"
    labelencoder = {"control": 0, "stimulated": 1}
    cell_type_key = "cell_type"
    condition_key = "condition"
    train_adata = sc.read(train_path)
    test_adata=sc.read(test_path)
    # if data.shape[1] > 2000:
    #     sc.pp.highly_variable_genes(data, n_top_genes=2000)
    #     train_adata = data[:, data.var['highly_variable']]
    #     test_adata = test_adata[:, data.var['highly_variable']]
    print(test_adata)

    
    
    print("process for" + specific_celltype)
    net_train_adata = train_adata[
        ~((train_adata.obs[cell_type_key] == specific_celltype) & (
            train_adata.obs[condition_key].isin([target_condition])))]
    print(net_train_adata)
    network = trvae.models.trVAE(x_dimension=net_train_adata.shape[1],
                                 z_dimension=40,
                                 conditions=conditions,
                                 model_path=f"./models/trVAE/{data_name}_{specific_celltype}/",
                                 output_activation='relu',
                                 verbose=5
                                 )

    network.train(net_train_adata,
                  condition_key,
                  n_epochs=10000,
                  batch_size=512,
                  verbose=2,
                  early_stop_limit=20,
                  lr_reducer=10,
                  )
    network.model_to_use=f"./models/trVAE/{data_name}_{specific_celltype}/"

    cell_type_adata = test_adata[test_adata.obs[cell_type_key] == specific_celltype]
    source_adata = cell_type_adata[cell_type_adata.obs[condition_key] == source_condition]
    target_adata = cell_type_adata[cell_type_adata.obs[condition_key] == target_condition]
    source_labels = np.zeros(source_adata.shape[0]) + labelencoder[source_condition]
    target_labels = np.zeros(source_adata.shape[0]) + labelencoder[target_condition]
    # target_condition = source_adata.obs[condition_key].value_counts().index[0]
    pred_adata = network.predict(source_adata,
                                  condition_key,
                                  target_condition=target_condition
                                  )
    pred_adata.obs[condition_key] = [f"pred_perturbed"] * pred_adata.shape[0]
    pred_adata.obs[cell_type_key] = specific_celltype

    all_data=target_adata.concatenate(pred_adata)
    x = np.mean(target_adata.X, axis=0)
    y = np.mean(pred_adata.X, axis=0)
    m, b, r_value, p_value, std_err = stats.linregress(x, y)
    print(specific_celltype)
    print(r_value ** 2)
    all_data.write_h5ad(f"./data/reconstructed/trVAE/trvae{data_name}_{specific_celltype}_6000.h5ad")