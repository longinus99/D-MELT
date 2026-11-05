# D-MELT: Dirichlet process Mixture-guided Embedding Learning via Two-stage training for scRNA-seq Clustering 🧊


## 🧊 Model overview


D-MELT is a two-stage deep clustering framework for scRNA-seq data that combines contrastive representation learning with a Dirichlet process mixture model (DPMM). It learns cell embeddings while accounting for similarities between cells and infers the number of clusters from the learned representations.


1. **Warm-up stage.** Two augmented views of each cell are processed by a MoCo-based encoder with instance and cluster projection heads. Instance-level contrastive learning accounts for potential false negatives, while cluster-level learning encourages consistent assignments across views.
2. **Adaptive clustering stage.** The pretrained encoder and instance head are transferred to the second stage. A DPMM estimates mixture components in the embedding space, and component-level learning is combined with instance-level contrastive learning to refine the representations and clustering structure.


<img width="3542" height="1878" alt="Fig" src="https://github.com/user-attachments/assets/c30ce74e-cb3c-4e5a-83ca-0b698e3ad00e" />


## 🧰 Input format


We recommend one AnnData (`.h5ad`) file per dataset, with cells as rows and genes as columns.


| Field | Content |
| --- | --- |
| `adata.X` | Raw gene expression counts, **cells × genes**; preprocessing is applied by the loader. |
| `adata.obs["cell_type"]` | Reference cell-type annotations required by the current training script for evaluation. Use `--label_key` to select another column. |


The loader also supports the following formats:


| Format | Expression matrix | Reference labels |
| --- | --- | --- |
| `.h5ad` (recommended) | `adata.X` | `adata.obs[args.label_key]` (default: `cell_type`) |
| `.h5` | `X` | `Y` |
| `.mat` | `feature` | `label` |


For `.h5` and `.mat` inputs, add the dataset name (without its extension) to `h5_datasets` or `mat_datasets` in `model/main.py`.


### Outputs


`train.run(...)` returns a dictionary with the following entries:


| Key | Content |
| --- | --- |
| `results["features"]` | Learned cell embeddings. |
| `results["q_instance"]` | Instance embeddings (the same embeddings as `features`). |
| `results["q_cluster"]` | Soft cluster assignments; use `argmax(axis=1)` to obtain cell labels. |
| `results["nmi"]`, `results["ari"]` | Clustering scores against the reference annotations. |


To save embeddings and assignments in an `.h5ad` file, enable the commented AnnData export block at the end of `model/main.py`.


## 📦 Installation
```bash
git clone https://github.com/longinus99/D-MELT.git
cd D-MELT
conda env create -f environment.yml
conda activate D-MELT
cd model
```
## 🚀 Training D-MELT


1. Save your dataset as `<dataset_name>.h5ad` and set `path` in `model/main.py` to the directory containing the file.
2. For `.h5` or `.mat` files, register the dataset name in the corresponding list as described above.
3. Run the training script from the `model/` directory:


```bash
cd model
python main.py --name 10X_PBMC
```


Replace `10X_PBMC` with your dataset name without the extension, and set `--label_key` to the annotation column in `adata.obs`.


## Recommended Hyperparameters of D-MELT
Many settings for training and model configuration are managed in `config.py`. This includes parameters such as the dataset name, number of training epochs, number of genes to select (`--select_gene`), learning rate, dropout rate, and more. For example, the following parameters are defined:


- `--name`: Dataset name (default: 10X_PBMC)
- `--cuda`: Cuda (default: 0)
- `--epoch`: Number of training epochs (default: 200)
- `--dropout`: Dropout rate (default: 0.9)
- `--temperature`: 


## 🗂️ Repository
