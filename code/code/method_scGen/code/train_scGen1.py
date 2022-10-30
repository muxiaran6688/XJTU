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
                                            # cell_type = "Neutrophil",
                                            cell_type_to_train=None):
    if data_name == "pbmc":
        stim_key = "stimulated"
        ctrl_key = "control"
        cell_type_key = "cell_type"

    elif data_name == "kang":
        stim_key = "stimulated"
        ctrl_key = "control"
        cell_type_key = "cell_type"
        train = sc.read("../data/train_kang.h5ad")
        valid = sc.read("../data/valid_kang.h5ad")
    elif data_name == "salmonella":
        stim_key = "Salmonella"
        ctrl_key = "Control"
        cell_type_key = "cell_label"
        train = sc.read("../data/train_salmonella.h5ad")
        valid = sc.read("../data/valid_salmonella.h5ad")
    elif data_name == "species":
        stim_key = "LPS6"
        ctrl_key = "unst"
        cell_type_key = "species"
        train = sc.read("../data/train_species.h5ad")
        valid = sc.read("../data/valid_species.h5ad")
    elif data_name == "covid":
        stim_key = "severe COVID-19"
        ctrl_key = "control"
        cell_type_key = "celltype"
        train = sc.read(f"../data/train_covid.h5ad")
        valid = sc.read(f"../data/valid_covid.h5ad")

    # abc=["Macrophages","Neutrophil", "NK","Secretory","mDC","CD8 T"]
    abc=["CD14+Mono","NK","Dendritic"]
    # abc=['CD14 Mono', 'CD4 T', 'T', 'CD8 T', 'B', 'DC', 'NK']
    # for idx, cell_type in enumerate(train.obs[cell_type_key].unique().tolist()):
    for cell_type in abc:
        if cell_type_to_train is not None and cell_type != cell_type_to_train:
            continue
        train = sc.read(f"../data/train_pbmc_{cell_type}.h5ad")
        valid = sc.read(f"../data/valid_pbmc_{cell_type}.h5ad")
        net_train_data = train[train.obs[cell_type_key] == cell_type]
        net_valid_data = valid[valid.obs[cell_type_key] == cell_type]
        network = scgen.VAEArith(x_dimension=net_train_data.X.shape[1],
                                  z_dimension=z_dim,
                                  alpha=alpha,
                                  dropout_rate=dropout_rate,
                                  learning_rate=learning_rate,
                                  model_path=f"../models/scGen/{data_name}/{cell_type}/scgen")

        network.train(net_train_data, use_validation=True, valid_data=net_valid_data, n_epochs=n_epochs, batch_size=batch_size, 
                      early_stop_limit=20)
        network.sess.close()
        print(f"network_{cell_type} has been trained!")


def reconstruct_whole_data(data_name="pbmc", condition_key="condition"):
    if data_name == "pbmc":
        stim_key = "stimulated"
        ctrl_key = "control"
        cell_type_key = "cell_type"

    elif data_name == "kang":
        stim_key = "stimulated"
        ctrl_key = "control"
        cell_type_key = "cell_type"
        train = sc.read("../data/train_kang.h5ad")
        test = sc.read("../data/test_kang.h5ad")

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
        train = sc.read("../data/train_covid.h5ad")
        test = sc.read("../data/test_covid.h5ad")


    all_data = anndata.AnnData()
    # test_data = sc.read(f"../data/test_covid_{cell_type}.h5ad")
    # for idx, cell_type in enumerate(train.obs[cell_type_key].unique().tolist()):
    # abc=['CD14 Mono', 'CD4 T', 'T', 'CD8 T', 'B', 'DC', 'NK']
    # abc=["FCGR3A+Mono","CD4T", "B","CD14+Mono","NK","CD8T","Dendritic"]
    abc=["CD14+Mono","NK","Dendritic"]
    for idx, cell_type in enumerate(abc):
        train = sc.read(f"../data/train_pbmc_{cell_type}.h5ad")
        test = sc.read(f"../data/test_pbmc_{cell_type}.h5ad")
        print(f"Reconstructing for {cell_type}")
        network = scgen.VAEArith(x_dimension=train.X.shape[1],
                                 z_dimension=100,
                                 alpha=0.00005,
                                 dropout_rate=0.2,
                                 learning_rate=0.01,
                                 model_path=f"../models/scGen/{data_name}/{cell_type}/scgen")
        network.restore_model()
        
        
        cell_type_data = test[test.obs[cell_type_key] == cell_type]
        cell_type_ctrl_data = test[((test.obs[cell_type_key] == cell_type) & (test.obs[condition_key] == ctrl_key))]
        net_train_data = train[~((train.obs[cell_type_key] == cell_type) & (train.obs[condition_key] == stim_key))]
        train_data = train[train.obs[cell_type_key] == cell_type]
        pred, delta = network.predict(adata=train_data,
                                      adata1=cell_type_data,
                                      conditions={"ctrl": ctrl_key, "stim": stim_key},
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
        # if idx==0:
        #     all_data = ctrl_adata.concatenate(pred_adata, real_stim_adata)
        # else:
        #     all_data = all_data.concatenate(ctrl_adata, pred_adata, real_stim_adata)
        x = numpy.average(real_stim, axis=0)
        y = numpy.average(pred_adata.X, axis=0)
        m, b, r_value, p_value, std_err = stats.linregress(x, y)
        print(r_value ** 2)
        # all_data.write_h5ad(f"../data/reconstructed/scGenyuanshi/scgen{data_name}_{cell_type}.h5ad")
        # if idx == 0:
        #     all_data = ctrl_adata.concatenate(pred_adata, real_stim_adata)
        # else:
        #     all_data = all_data.concatenate(ctrl_adata, pred_adata, real_stim_adata)

        print(f"Finish Reconstructing for {cell_type}")
        network.sess.close()
        all_data = ctrl_adata.concatenate(pred_adata, real_stim_adata)
        all_data.write_h5ad(f"../data/reconstructed/scGen/scgen_{data_name}_{cell_type}.h5ad")


# def test_train_whole_data_some_celltypes_out(data_name="pbmc",
#                                              z_dim=100,
#                                              alpha=0.00005,
#                                              n_epochs=300,
#                                              batch_size=32,
#                                              dropout_rate=0.2,
#                                              learning_rate=0.001,
#                                              condition_key="condition",
#                                              c_out=None,
#                                              c_in=None):
#     if data_name == "pbmc":
#         stim_key = "stimulated"
#         ctrl_key = "control"
#         cell_type_key = "cell_type"
#         train = sc.read("../data/train_pbmc.h5ad")
#         valid = sc.read("../data/valid_pbmc.h5ad")

#     net_train_data = scgen.data_remover(train, remain_list=c_in, remove_list=c_out,
#                                         cell_type_key=cell_type_key, condition_key=condition_key)

#     net_valid_data = scgen.data_remover(valid, remain_list=c_in, remove_list=c_out,
#                                         cell_type_key=cell_type_key, condition_key=condition_key)

#     network = scgen.VAEArith(x_dimension=net_train_data.X.shape[1],
#                              z_dimension=z_dim,
#                              alpha=alpha,
#                              dropout_rate=dropout_rate,
#                              learning_rate=learning_rate,
#                              model_path=f"../models/scGen/pbmc/heldout/{len(c_out)}/scgen")

#     network.train(net_train_data, use_validation=True, valid_data=net_valid_data, n_epochs=n_epochs, batch_size=batch_size)
#     print(f"network has been trained!")
#     network.sess.close()


# def train_cross_study(data_name="study",
#                       z_dim=100,
#                       alpha=0.00005,
#                       n_epochs=300,
#                       batch_size=32,
#                       dropout_rate=0.2,
#                       learning_rate=0.001):
#     train = sc.read("../data/train_study.h5ad")
#     valid = sc.read("../data/valid_study.h5ad")

#     net_train_data = train
#     network = scgen.VAEArith(x_dimension=net_train_data.X.shape[1],
#                              z_dimension=z_dim,
#                              alpha=alpha,
#                              dropout_rate=dropout_rate,
#                              learning_rate=learning_rate,
#                              model_path="../models/scGen/study/scgen")

#     network.train(net_train_data, use_validation=True, valid_data=valid, n_epochs=n_epochs, batch_size=batch_size)
#     print(f"network_{data_name} has been trained!")
#     network.sess.close()


if __name__ == '__main__':
    # test_train_whole_data_one_celltype_out("covid", z_dim=100, alpha=0.00005, n_epochs=300, 
    #                                         batch_size=32, dropout_rate=0.2, learning_rate=0.001)
    # test_train_whole_data_one_celltype_out("covid", z_dim=100, alpha=0.00005, n_epochs=300, cell_type = "Macrophages",
    #                                        batch_size=32, dropout_rate=0.2, learning_rate=0.001)
    # test_train_whole_data_one_celltype_out("pbmc", z_dim=100, alpha=0.00005, n_epochs=5, batch_size=32,
                                            # dropout_rate=0.2, learning_rate=0.001)
    test_train_whole_data_one_celltype_out("pbmc", z_dim=100, alpha=0.00005, n_epochs=1, batch_size=32,
                                            dropout_rate=0.2, learning_rate=0.001)
    # test_train_whol                 e_data_one_celltype_out("hpoly", z_dim=100, alpha=0.00005, n_epochs=300, batch_size=32,
    #                                        dropout_rate=0.2, learning_rate=0.001)
    # test_train_whole_data_one_celltype_out("salmonella", z_dim=100, alpha=0.00005, n_epochs=300, batch_size=32,
    #                                        dropout_rate=0.2, learning_rate=0.001)
    # test_train_whole_data_one_celltype_out("species", z_dim=100, alpha=0.00005, n_epochs=300, batch_size=32,
    #                                        dropout_rate=0.2, learning_rate=0.001, cell_type_to_train="rat")
    # test_train_whole_data_one_celltype_out("pbmc", z_dim=100, alpha=0.00005, n_epochs=300, batch_size=32,
    #                                           dropout_rate=0.2, learning_rate=0.001, cell_type = "CD8T")
    # test_train_whole_data_one_celltype_out("pbmc", z_dim=100, alpha=0.001, n_epochs=300, batch_size=32,
    #                                           dropout_rate=0.2, learning_rate=0.001, cell_type = "CD14+Mono")
    # test_train_whole_data_one_celltype_out("pbmc", z_dim=100, alpha=0.00005, n_epochs=300, batch_size=32,
    #                                           dropout_rate=0.2, learning_rate=0.001, cell_type = "FCGR3A+Mono")
    # test_train_whole_data_one_celltype_out("pbmc", z_dim=100, alpha=0.001, n_epochs=300, batch_size=32,
    #                                           dropout_rate=0.2, learning_rate=0.001, cell_type = "Dendritic")
    # test_train_whole_data_one_celltype_out("pbmc", z_dim=100, alpha=0.00005, n_epochs=300, batch_size=32,
    #                                           dropout_rate=0.2, learning_rate=0.001, cell_type = "CD4T")
    # test_train_whole_data_one_celltype_out("pbmc", z_dim=100, alpha=0.00005, n_epochs=300, batch_size=32,
    #                                           dropout_rate=0.2, learning_rate=0.001, cell_type = "B")
    # test_train_whole_data_one_celltype_out("pbmc", z_dim=100, alpha=0.001, n_epochs=300, batch_size=32,
    #                                           dropout_rate=0.2, learning_rate=0.001, cell_type = "NK")
    # train_cross_study("study", z_dim=100, alpha=0.00005, n_epochs=300, batch_size=32,
    #                   dropout_rate=0.2, learning_rate=0.001)
    # reconstruct_whole_data("covid")
    # reconstruct_whole_data("covid", cell_type = "Macrophages")
    # reconstruct_whole_data("pbmc", cell_type = "CD8T")
    # reconstruct_whole_data("pbmc", cell_type = "CD14+Mono")
    # reconstruct_whole_data("pbmc", cell_type = "FCGR3A+Mono")
    # reconstruct_whole_data("pbmc", cell_type = "Dendritic")
    reconstruct_whole_data("pbmc")
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
