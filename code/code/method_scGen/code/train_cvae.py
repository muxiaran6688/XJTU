import scgen
import scanpy as sc
import numpy as np

a=["CD8T","NK","Dendritic","CD14+Mono","FCGR3A+Mono"]
for cell_type in a:
    train = sc.read(f"../data/train_pbmc_{cell_type}.h5ad")
    valid = sc.read(f"../data/valid_pbmc_{cell_type}.h5ad")
    train = train[~((train.obs["cell_type"] == cell_type) & (train.obs["condition"] == "stimulated"))]
    valid = valid[~((valid.obs["cell_type"] == cell_type) & (valid.obs["condition"] == "stimulated"))]
    z_dim = 20
    network = scgen.CVAE(x_dimension=train.X.shape[1], z_dimension=z_dim, alpha=0.1, model_path=f"../models/CVAE/pbmc/{cell_type}/models/scgen")
    network.train(train, use_validation=True, valid_data=valid, n_epochs=100)
    labels, _ = scgen.label_encoder(train)
    train = sc.read(f"../data/test_pbmc_{cell_type}.h5ad")
    CD4T = train[train.obs["cell_type"] == cell_type]
    unperturbed_data = train[((train.obs["cell_type"] == cell_type) & (train.obs["condition"] == "control"))]
    fake_labels = np.ones((len(unperturbed_data), 1))
    predicted_cells = network.predict(unperturbed_data, fake_labels)
    adata = sc.AnnData(predicted_cells, obs={"condition": ["pred"]*len(fake_labels)})
    adata.var_names = CD4T.var_names
    all_adata = CD4T.concatenate(adata)
    all_adata.write(f"../data/reconstructed/CVAE_{cell_type}.h5ad")