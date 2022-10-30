import scgen
import scanpy as sc
import numpy as np
from scipy import sparse
from scipy import stats

cell_type_list=["Secretory","NK","mDC","Macrophages","CD8 T","Neutrophil"]

data_name="covid"
train = sc.read(f"/home/datasets/psvae/train_{data_name}.h5ad")
valid = sc.read(f"/home/datasets/psvae/valid_{data_name}.h5ad")
test = sc.read(f"/home/datasets/psvae/test_{data_name}.h5ad")
for cell_type in cell_type_list:
    

    train=train[~((train.obs['celltype']==cell_type)&(train.obs['condition']=="severe COVID-19"))]
    valid=valid[~((valid.obs['celltype']==cell_type)&(valid.obs['condition']=="severe COVID-19"))]
    test_adata = test[(test.obs["condition"] == "control")&(test.obs["celltype"]==cell_type)]
    
    z_dim = 20
    network = scgen.CVAE(x_dimension=train.X.shape[1], z_dimension=z_dim, alpha=0.01, model_path=f"../models/CVAE/{data_name}/{cell_type}/models/scgen")
    network.train(train, use_validation=True, valid_data=valid, n_epochs=100)
    labels, _ = scgen.label_encoder(test_adata)

    # CD4T = train[train.obs["cell_type"] == "CD4T"]

    fake_labels = np.ones((len(test_adata), 1))
    predicted_cells = network.predict(test_adata, fake_labels)
    adata = sc.AnnData(predicted_cells, obs={"condition": [f"{cell_type}_pred_pert"]*len(fake_labels)})
    adata.var_names = test.var_names
    all_adata = test.concatenate(adata)
    all_adata.obs["condition"].replace("control", f"{cell_type}_ctrl", inplace=True)
    all_adata.obs["condition"].replace("severe COVID-19", f"{cell_type}_real_pert", inplace=True)
    if sparse.issparse(test.X):
        real_sev=test[test.obs["condition"]=="severe COVID-19"].X.A
    else:
        real_sev=test[test.obs["condition"]=="severe COVID-19"].X
    x = np.average(real_sev, axis=0)
    y = np.average(adata.X, axis=0)
    m, b, r_value, p_value, std_err = stats.linregress(x, y)
    print(cell_type)
    print(r_value ** 2)
    all_adata.write(f"/home/datasets/psvae/reconstructed/CVAE/CVAE{data_name}_{cell_type}.h5ad")