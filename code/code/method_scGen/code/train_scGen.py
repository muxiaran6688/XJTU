# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 21:01:56 2022

@author: 70772
"""

import anndata
import scanpy as sc
import numpy
import scgen
from scipy import sparse
from scipy import stats

def test_train_whole_data_one_celltype_out(data_name="pbmc",
                                           z_dim=50,
                                           alpha=0.1,
                                           n_epochs=1000,
                                           batch_size=32,
                                           dropout_rate=0.25,
                                           learning_rate=0.001,
                                           condition_key="condition",
                                           cell_type = "Neutrophil",
                                           cell_type_to_train=None):
    if data_name == "covid":
        stim_key = "severe COVID-19"
        ctrl_key = "control"
        # cell_type = "Neutrophil"
        # train0 = sc.read("../data/control_train_pbmc_CD4T.h5ad")
        # train1 = sc.read("../data/stimulated_train_pbmc_CD4T.h5ad")
        # valid0 = sc.read("../data/control_valid_pbmc_CD4T.h5ad")
        # valid1 = sc.read("../data/stimulated_valid_pbmc_CD4T.h5ad")
        train=sc.read(f"../data/train_{data_name}_{cell_type}.h5ad")
        valid=sc.read(f"../data/valid_{data_name}_{cell_type}.h5ad")
        # sc.pp.highly_variable_genes(train, n_top_genes=2000)
        # train = train[:, train.var['highly_variable']]
        # valid=valid[:, train.var['highly_variable']]
        train0=train[train.obs['condition']=="control"]
        train1=train[train.obs['condition']=="severe COVID-19"]
        valid0=valid[valid.obs['condition']=="control"]
        valid1=valid[valid.obs['condition']=="severe COVID-19"]

    if data_name == "pbmc":
        stim_key = "stimulated"
        ctrl_key = "control"
        # cell_type = "Neutrophil"
        # train0 = sc.read("../data/control_train_pbmc_CD4T.h5ad")
        # train1 = sc.read("../data/stimulated_train_pbmc_CD4T.h5ad")
        # valid0 = sc.read("../data/control_valid_pbmc_CD4T.h5ad")
        # valid1 = sc.read("../data/stimulated_valid_pbmc_CD4T.h5ad")
        train=sc.read(f"../data/train_{data_name}_{cell_type}.h5ad")
        valid=sc.read(f"../data/valid_{data_name}_{cell_type}.h5ad")
        train0=train[train.obs['condition']=="control"]
        train1=train[train.obs['condition']=="stimulated"]
        valid0=valid[valid.obs['condition']=="control"]
        valid1=valid[valid.obs['condition']=="stimulated"]
        
    network = scgen.VAEArith(x_dimension=train0.X.shape[1],
                             z_dimension=z_dim,
                             alpha=alpha,
                             dropout_rate=dropout_rate,
                             learning_rate=learning_rate,
                             model_path=f"../models/scGenmymethod/{data_name}/{cell_type}/scgen")

    network.train(train_data=train0, train_data1=train1, use_validation=True, valid_data=valid0, valid_data1=valid1, n_epochs=n_epochs, batch_size=batch_size)
    print("network has been trained!")
    network.sess.close()
    # print(f"network_{cell_type} has been trained!")

def reconstruct_whole_data(data_name="pbmc", condition_key="condition", cell_type = "Neutrophil"):
    if data_name == "pbmc":
        stim_key = "stimulated"
        ctrl_key = "control"
        cell_type_key = "cell_type"
        train = sc.read(f"../data/train_pbmc_{cell_type}.h5ad")
        test_data = sc.read(f"../data/test_pbmc_{cell_type}.h5ad")
    elif data_name == "hpoly":
        stim_key = "Hpoly.Day10"
        ctrl_key = "Control"
        cell_type_key = "cell_label"
        train = sc.read("../data/train_hpoly.h5ad")
    elif data_name == "salmonella":
        stim_key = "Salmonella"
        ctrl_key = "Control"
        cell_type_key = "cell_label"
        train = sc.read("../data/train_salmonella.h5ad")
    elif data_name == "species":
        stim_key = "LPS6"
        ctrl_key = "unst"
        cell_type_key = "species"
        train = sc.read("../data/train_species.h5ad")
    elif data_name == "study":
        stim_key = "stimulated"
        ctrl_key = "control"
        cell_type_key = "cell_type"
        train = sc.read("../data/train_study.h5ad")
    elif data_name == "covid":
        stim_key = "severe COVID-19"
        ctrl_key = "control"
        cell_type_key = "celltype"
        # cell_type = "Neutrophil"
        train = sc.read(f"../data/train_covid_{cell_type}.h5ad")
        test_data = sc.read(f"../data/test_covid_{cell_type}.h5ad")
        # sc.pp.highly_variable_genes(train, n_top_genes=2000)
        # train = train[:, train.var['highly_variable']]
        # test_data=test_data[:, train.var['highly_variable']]


    all_data = anndata.AnnData()
    # test_data = sc.read(f"../data/test_covid_{cell_type}.h5ad")
    for idx, cell_type in enumerate(train.obs[cell_type_key].unique().tolist()):
        print(f"Reconstructing for {cell_type}")
        network = scgen.VAEArith(x_dimension=train.X.shape[1],
                                 z_dimension=100,
                                 alpha=0.00005,
                                 dropout_rate=0.2,
                                 learning_rate=0.001,
                                 model_path=f"../models/scGenmymethod/{data_name}/{cell_type}/scgen")
        network.restore_model()

        cell_type_data = test_data
        cell_type_ctrl_data = test_data[((test_data.obs[cell_type_key] == cell_type) & (test_data.obs[condition_key] == ctrl_key))]
        net_train_data = train[~((train.obs[cell_type_key] == cell_type) & (train.obs[condition_key] == stim_key))]
        pred, delta = network.predict(adata=train,
                                      adata1=test_data,
                                      conditions = {"ctrl": ctrl_key, "stim": stim_key},
                                      cell_type_key=cell_type_key,
                                      condition_key=condition_key,
                                      celltype_to_predict=cell_type,
                                      biased=True)

        pred_adata = anndata.AnnData(pred, obs={condition_key: [f"{cell_type}_pred_sev"] * len(pred),
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
                                          obs={condition_key: [f"{cell_type}_real_sev"] * len(real_stim),
                                               cell_type_key: [cell_type] * len(real_stim)},
                                          var={"var_names": cell_type_data.var_names})
        if idx == 0:
            all_data = ctrl_adata.concatenate(pred_adata, real_stim_adata)
        else:
            all_data = all_data.concatenate(ctrl_adata, pred_adata, real_stim_adata)

        print(f"Finish Reconstructing for {cell_type}")
        x = numpy.average(real_stim, axis=0)
        y = numpy.average(pred_adata.X, axis=0)
        m, b, r_value, p_value, std_err = stats.linregress(x, y)
        print(r_value ** 2)
        network.sess.close()
    # all_data.write_h5ad(f"../data/reconstructed/PSVAE/{data_name}_{cell_type}.h5ad")
    if r_value ** 2>0.895:
        
        all_data.write_h5ad(f"../data/reconstructed/PSVAE/{data_name}_{cell_type}.h5ad")

if __name__ == '__main__':
    # test_train_whole_data_one_celltype_out("covid", z_dim=100, alpha=0.00005, n_epochs=280, batch_size=32,
    #                                         dropout_rate=0.2, learning_rate=0.001, cell_type = "NK")
    # test_train_whole_data_one_celltype_out("covid", z_dim=100, alpha=0.00005, n_epochs=280, batch_size=32,
    #                                         dropout_rate=0.2, learning_rate=0.001, cell_type = "mDC")
    # test_train_whole_data_one_celltype_out("covid", z_dim=100, alpha=0.00005, n_epochs=300, batch_size=32,
    #                                         dropout_rate=0.2, learning_rate=0.001, cell_type = "Macrophages")
    test_train_whole_data_one_celltype_out("covid", z_dim=100, alpha=0.00005, n_epochs=320, batch_size=32,
                                            dropout_rate=0.2, learning_rate=0.001, cell_type = "Neutrophil")
    # test_train_whole_data_one_celltype_out("hpoly", z_dim=100, alpha=0.00005, n_epochs=300, batch_size=32,
    #                                        dropout_rate=0.2, learning_rate=0.001)
    # test_train_whole_data_one_celltype_out("pbmc", z_dim=100, alpha=0.00005, n_epochs=300, batch_size=32,
    #                                           dropout_rate=0.2, learning_rate=0.001, cell_type = "CD8T")
    # test_train_whole_data_one_celltype_out("pbmc", z_dim=100, alpha=0.00005, n_epochs=300, batch_size=32,
    #                                           dropout_rate=0.2, learning_rate=0.001, cell_type = "CD14+Mono")
    # test_train_whole_data_one_celltype_out("pbmc", z_dim=100, alpha=0.00005, n_epochs=300, batch_size=32,
    #                                           dropout_rate=0.2, learning_rate=0.001, cell_type = "FCGR3A+Mono")
    # test_train_whole_data_one_celltype_out("pbmc", z_dim=100, alpha=0.00005, n_epochs=300, batch_size=32,
    #                                           dropout_rate=0.2, learning_rate=0.001, cell_type = "Dendritic")
    # test_train_whole_data_one_celltype_out("pbmc", z_dim=100, alpha=0.00005, n_epochs=280, batch_size=32,
    #                                          dropout_rate=0.2, learning_rate=0.001, cell_type = "fmono")
    # test_train_whole_data_one_celltype_out("pbmc", z_dim=100, alpha=0.00005, n_epochs=300, batch_size=32,
    #                                           dropout_rate=0.2, learning_rate=0.001, cell_type = "CD4T")
    # test_train_whole_data_one_celltype_out("pbmc", z_dim=100, alpha=0.00005, n_epochs=300, batch_size=32,
    #                                           dropout_rate=0.2, learning_rate=0.001, cell_type = "B")
    # test_train_whole_data_one_celltype_out("pbmc", z_dim=100, alpha=0.00005, n_epochs=300, batch_size=32,
    #                                           dropout_rate=0.2, learning_rate=0.001, cell_type = "NK")
    # test_train_whole_data_one_celltype_out("salmonella", z_dim=100, alpha=0.00005, n_epochs=300, batch_size=32,
    #                                        dropout_rate=0.2, learning_rate=0.001)
    # test_train_whole_data_one_celltype_out("species", z_dim=100, alpha=0.00005, n_epochs=300, batch_size=32,
    #                                        dropout_rate=0.2, learning_rate=0.001, cell_type_to_train="rat")
    # train_cross_study("study", z_dim=100, alpha=0.00005, n_epochs=300, batch_size=32,
    #                   dropout_rate=0.2, learning_rate=0.001)
    # reconstruct_whole_data("covid", cell_type = "NK")
    # reconstruct_whole_data("covid", cell_type = "Secretory")
    # reconstruct_whole_data("covid", cell_type = "Macrophages")
    reconstruct_whole_data("covid", cell_type = "Neutrophil")
    # reconstruct_whole_data("covid", cell_type = "mDC")
    # reconstruct_whole_data("pbmc", cell_type = "CD14+Mono")
    # reconstruct_whole_data("pbmc", cell_type = "FCGR3A+Mono")
    # reconstruct_whole_data("pbmc", cell_type = "Dendritic")
    # reconstruct_whole_data("pbmc", cell_type = "fmono")
    # reconstruct_whole_data("pbmc", cell_type = "CD4T")
    # reconstruct_whole_data("pbmc", cell_type = "B")
    # reconstruct_whole_data("pbmc", cell_type = "NK")
    # reconstruct_whole_data("hpoly")
    # reconstruct_whole_data("salmonella")
    # reconstruct_whole_data("species")

    # c_in = ['NK', 'B', 'CD14+Mono']
    # c_out = ['CD4T', 'FCGR3A+Mono', 'CD8T', 'Dendritic']
    # test_train_whole_data_some_celltypes_out(data_name="pbmc",
    #                                          z_dim=100,
    #                                          alpha=0.00005,
    #                                          n_epochs=300,
    #                                          batch_size=32,
    #                                          dropout_rate=0.2,
    #                                          learning_rate=0.001,
    #                                          condition_key="condition",
    #                                          c_out=c_out,
    #                                          c_in=c_in)
    # c_in = ['CD14+Mono']
    # c_out = ['CD4T', 'FCGR3A+Mono', 'CD8T', 'NK', 'B', 'Dendritic']
    # test_train_whole_data_some_celltypes_out(data_name="pbmc",
    #                                          z_dim=100,
    #                                          alpha=0.00005,
    #                                          n_epochs=300,
    #                                          batch_size=32,
    #                                          dropout_rate=0.2,
    #                                          learning_rate=0.001,
    #                                          condition_key="condition",
    #                                          c_out=c_out,
    #                                          c_in=c_in)
    # c_in = ['CD8T', 'NK', 'B', 'Dendritic', 'CD14+Mono']
    # c_out = ['CD4T', 'FCGR3A+Mono']
    # test_train_whole_data_some_celltypes_out(data_name="pbmc",
    #                                          z_dim=100,
    #                                          alpha=0.00005,
    #                                          n_epochs=300,
    #                                          batch_size=32,
    #                                          dropout_rate=0.2,
    #                                          learning_rate=0.001,
    #                                          condition_key="condition",
    #                                          c_out=c_out,
    #                                          c_in=c_in)