# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 21:01:56 2022

@author: 70772
"""

import anndata
import scanpy as sc
import numpy
import psvae
from scipy import sparse
from scipy import stats

def train(data_name="pbmc",z_dim=50,alpha=0.1,
                                    n_epochs=1000,
                                    batch_size=32,
                                    dropout_rate=0.25,
                                    learning_rate=0.001,
                                    condition_key="condition",
                                    cell_type = "CD4T",
                                    cell_type_to_train=None):
    if data_name == "covid":
        stim_key = "severe COVID-19"
        ctrl_key = "control"

        train=sc.read(f"/home/datasets/psvae/train_{data_name}.h5ad")
        valid=sc.read(f"/home/datasets/psvae/valid_{data_name}.h5ad")

        train0=train[((train.obs['condition']=="control")&(train.obs['celltype']==cell_type))]
        train1=train[((train.obs['condition']=="severe COVID-19")&(train.obs['celltype']==cell_type))]
        valid0=valid[((valid.obs['condition']=="control")&(valid.obs['celltype']==cell_type))]
        valid1=valid[((valid.obs['condition']=="severe COVID-19")&(valid.obs['celltype']==cell_type))]

    if data_name == "pbmc":
        stim_key = "stimulated"
        ctrl_key = "control"

        train=sc.read(f"./data/train_{data_name}_{cell_type}.h5ad")
        valid=sc.read(f"./data/valid_{data_name}_{cell_type}.h5ad")
        train0=train[train.obs['condition']=="control"]
        train1=train[train.obs['condition']=="stimulated"]
        valid0=valid[valid.obs['condition']=="control"]
        valid1=valid[valid.obs['condition']=="stimulated"]
        
    network = psvae.VAEArith(x_dimension=train0.X.shape[1],
                             z_dimension=z_dim,
                             alpha=alpha,
                             dropout_rate=dropout_rate,
                             learning_rate=learning_rate,
                             model_path=f"./models/psvae/{data_name}/{cell_type}/psvae")

    network.train(train_data=train0, train_data1=train1, use_validation=True, valid_data=valid0, valid_data1=valid1, n_epochs=n_epochs, batch_size=batch_size)
    print("network has been trained!")
    network.sess.close()
    # print(f"network_{cell_type} has been trained!")

def reconstruct(data_name="pbmc", condition_key="condition", cell_type = "CD4T"):
    if data_name == "pbmc":
        stim_key = "stimulated"
        ctrl_key = "control"
        cell_type_key = "cell_type"
        train = sc.read(f"./data/train_pbmc_{cell_type}.h5ad")
        test_data = sc.read(f"./data/test_pbmc_{cell_type}.h5ad")
    
    elif data_name == "covid":
        stim_key = "severe COVID-19"
        ctrl_key = "control"
        cell_type_key = "celltype"
        train_all = sc.read(f"/home/datasets/psvae/train_{data_name}.h5ad")
        test_all = sc.read(f"/home/datasets/psvae/test_{data_name}.h5ad")
        train = train_all[train_all.obs["celltype"]==cell_type]
        test_data = test_all[test_all.obs["celltype"]==cell_type]


    all_data = anndata.AnnData()


    print(f"Reconstructing for {cell_type}")
    network = psvae.VAEArith(x_dimension=train.X.shape[1],
                             z_dimension=100,
                             alpha=0.00005,
                             dropout_rate=0.2,
                             learning_rate=0.001,
                             model_path=f"./models/psvae/{data_name}/{cell_type}/psvae")
    network.restore_model()

    cell_type_data = test_data
    cell_type_ctrl_data = test_data[((test_data.obs[cell_type_key] == cell_type) & (test_data.obs[condition_key] == ctrl_key))]
        
    pred, delta = network.predict(adata=train,
                                  adata1=test_data,
                                  conditions = {"ctrl": ctrl_key, "stim": stim_key},
                                  cell_type_key=cell_type_key,
                                  condition_key=condition_key,
                                  celltype_to_predict=cell_type,
                                  biased=True)

    pred_adata = anndata.AnnData(pred, obs={condition_key: [f"{cell_type}_pred_pert"] * len(pred),
                                            cell_type_key: [cell_type] * len(pred)},
                                 var={"var_names": cell_type_data.var_names})
    ctrl_adata = anndata.AnnData(cell_type_ctrl_data.X,
                                 obs={condition_key: [f"{cell_type}_ctrl"] * len(cell_type_ctrl_data),
                                      cell_type_key: [cell_type] * len(cell_type_ctrl_data)},
                                 var={"var_names": cell_type_ctrl_data.var_names})
    if sparse.issparse(cell_type_data.X):
        real_stim = cell_type_data[cell_type_data.obs[condition_key] == stim_key].X.A
    else:
        real_stim = cell_type_data[cell_type_data.obs[condition_key] == stim_key].X
    real_stim_adata = anndata.AnnData(real_stim,
                                      obs={condition_key: [f"{cell_type}_real_pert"] * len(real_stim),
                                           cell_type_key: [cell_type] * len(real_stim)},
                                      var={"var_names": cell_type_data.var_names})

    all_data = ctrl_adata.concatenate(pred_adata, real_stim_adata)


    print(f"Finish Reconstructing for {cell_type}")
    x = numpy.average(real_stim, axis=0)
    y = numpy.average(pred_adata.X, axis=0)
    m, b, r_value, p_value, std_err = stats.linregress(x, y)
    print(r_value ** 2)
    network.sess.close()
    all_data.write_h5ad(f"/home/datasets/psvae/reconstructed/PSVAE/{data_name}_{cell_type}.h5ad")
    

if __name__ == '__main__':
    # train("covid", z_dim=100, alpha=0.00005, n_epochs=340, batch_size=32,
    #       dropout_rate=0.2, learning_rate=0.001, cell_type = "NK")
    # train("covid", z_dim=100, alpha=0.00005, n_epochs=280, batch_size=32,
    #       dropout_rate=0.2, learning_rate=0.001, cell_type = "Secretory")
    # train("covid", z_dim=100, alpha=0.00005, n_epochs=320, batch_size=32,
    #       dropout_rate=0.2, learning_rate=0.0001, cell_type = "Macrophages")
    # train("covid", z_dim=100, alpha=0.00005, n_epochs=280, batch_size=32,
    #       dropout_rate=0.2, learning_rate=0.001, cell_type = "Neutrophil")
    # train("covid", z_dim=100, alpha=0.00005, n_epochs=300, batch_size=32,
    #       dropout_rate=0.2, learning_rate=0.001, cell_type = "mDC")
    # train("covid", z_dim=100, alpha=0.00005, n_epochs=320, batch_size=32,
    #       dropout_rate=0.2, learning_rate=0.001, cell_type = "CD8 T")
    
    # train("pbmc", z_dim=100, alpha=0.00005, n_epochs=300, batch_size=32,
    #       dropout_rate=0.2, learning_rate=0.001, cell_type = "CD8T")
    # train("pbmc", z_dim=100, alpha=0.00005, n_epochs=300, batch_size=32,
    #       dropout_rate=0.2, learning_rate=0.001, cell_type = "CD14+Mono")
    # train("pbmc", z_dim=100, alpha=0.00005, n_epochs=300, batch_size=32,
    #       dropout_rate=0.2, learning_rate=0.001, cell_type = "FCGR3A+Mono")
    train("pbmc", z_dim=100, alpha=0.00005, n_epochs=350, batch_size=32,
          dropout_rate=0.2, learning_rate=0.001, cell_type = "Dendritic")

    # train("pbmc", z_dim=100, alpha=0.00005, n_epochs=300, batch_size=32,
    #       dropout_rate=0.2, learning_rate=0.001, cell_type = "CD4T")
    # train("pbmc", z_dim=100, alpha=0.00005, n_epochs=300, batch_size=32,
    #       dropout_rate=0.2, learning_rate=0.001, cell_type = "B")
    # train("pbmc", z_dim=100, alpha=0.00005, n_epochs=300, batch_size=32,
    #       dropout_rate=0.2, learning_rate=0.001, cell_type = "NK")
    
    # reconstruct("covid", cell_type = "NK")
    # reconstruct("covid", cell_type = "Secretory")
    # reconstruct("covid", cell_type = "Macrophages")
    # reconstruct("covid", cell_type = "Neutrophil")
    # reconstruct("covid", cell_type = "mDC")
    # reconstruct("covid", cell_type = "CD8 T")
    # reconstruct("pbmc", cell_type = "CD14+Mono")
    # reconstruct("pbmc", cell_type = "FCGR3A+Mono")
    reconstruct("pbmc", cell_type = "Dendritic")
    # reconstruct("pbmc", cell_type = "CD8T")
    # reconstruct("pbmc", cell_type = "CD4T")
    # reconstruct("pbmc", cell_type = "B")
    # reconstruct("pbmc", cell_type = "NK")

