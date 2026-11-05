<h1 align="center">🧊 D-MELT</h1>

<h3 align="center">Dirichlet Process Mixture-Guided Embedding Learning via Two-Stage Training for scRNA-seq Clustering</h3>

<p align="center"><em>Learn cell embeddings and infer the number of clusters directly from single-cell gene expression.</em></p>

<p align="center">
  <a href="environment.yml"><img alt="Python 3.7.12" src="https://img.shields.io/badge/python-3.7.12-blue"></a>
  <a href="environment.yml"><img alt="PyTorch 1.13.1" src="https://img.shields.io/badge/PyTorch-1.13.1-ee4c2c"></a>
</p>


## 🧊 Model overview


D-MELT is a two-stage deep clustering framework for scRNA-seq data that combines contrastive representation learning with a Dirichlet process mixture model (DPMM). It learns cell embeddings while accounting for similarities between cells and infers the number of clusters from the learned representations.


1. **Warm-up stage.** Two augmented views of each cell are processed by a MoCo-based encoder with instance and cluster projection heads. Instance-level contrastive learning accounts for potential false negatives, while cluster-level learning encourages consistent assignments across views.
2. **Adaptive clustering stage.** The pretrained encoder and instance head are transferred to the second stage. A DPMM estimates mixture components in the embedding space, and component-level learning is combined with instance-level contrastive learning to refine the representations and clustering structure.


<img width="3542" height="1878" alt="Fig" src="https://github.com/user-attachments/assets/c30ce74e-cb3c-4e5a-83ca-0b698e3ad00e" />


## 🗃️ Data

We evaluated D-MELT on 15 publicly available scRNA-seq datasets. Click a dataset name below to open its source page.

<table>
  <tr>
    <td><a href="https://www.10xgenomics.com/resources/datasets/3-k-pbm-cs-from-a-healthy-donor-1-standard-1-1-0">10X PBMC</a></td>
    <td><a href="https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE67835">Darmanis</a></td>
    <td><a href="https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE81252">Camp1</a></td>
  </tr>
  <tr>
    <td><a href="https://www.ebi.ac.uk/biostudies/arrayexpress/studies/E-MTAB-5061">Segerstolpe</a></td>
    <td><a href="https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE65525">Klein</a></td>
    <td><a href="https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE94333">Adam</a></td>
  </tr>
  <tr>
    <td><a href="https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE115978">Melanoma</a></td>
    <td><a href="https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE84133">Baron</a></td>
    <td><a href="https://singlecell.broadinstitute.org/single_cell/study/SCP345">Slyper</a></td>
  </tr>
  <tr>
    <td><a href="https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE87544">Chen</a></td>
    <td><a href="https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE124952">Bhattacherjee</a></td>
    <td><a href="https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE189120">Ghanem</a></td>
  </tr>
  <tr>
    <td><a href="https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE127465">Zilionis</a></td>
    <td><a href="https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE181919">HNSCC</a></td>
    <td><a href="https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE135133">Orozco</a></td>
  </tr>
</table>

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

Create and activate the Conda environment using [`environment.yml`](environment.yml) to install the dependencies.

```bash
git clone https://github.com/longinus99/D-MELT.git
cd D-MELT
conda env create -f environment.yml
conda activate D-MELT
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


## ⚙️ Recommended Hyperparameters of D-MELT
Many settings for training and model configuration are managed in `config.py`. This includes parameters such as the dataset name, number of training epochs, number of genes to select (`--select_gene`), learning rate, dropout rate, and more. For example, the following parameters are defined:


- `--name`: Dataset name (default: 10X_PBMC)
- `--cuda`: Cuda (default: 0)
- `--epoch`: Number of training epochs (default: 200)
- `--dropout`: Dropout rate (default: 0.9)
- `--temperature` (default: `0.07`) scales similarity logits in the instance-level contrastive loss. Lower values make the softmax distribution sharper.
- `--phi` (default: `0.95`) sets the similarity threshold for filtering potential false negatives in the instance-level contrastive loss.


## 🎛️ Dataset-specific parameter settings

HNSCC used `--dropout 0.95`. All other datasets used the default dropout.

| Parameter | 10X PBMC | Darmanis | Camp1 | Segerstolpe | Klein |
| --- | --- | --- | --- | --- | --- |
| `--temperature` | 0.10 | 0.07 | 0.06 | 0.10 | 0.05 |

| Parameter | Adam | Melanoma | Baron | Slyper | Chen |
| --- | --- | --- | --- | --- | --- |
| `--temperature` | 0.10 | 0.70 | 0.70 | 0.90 | 0.50 |

| Parameter | Bhattacherjee | Ghanem | Zilionis | HNSCC | Orozco |
| --- | --- | --- | --- | --- | --- |
| `--temperature` | 1.00 | 0.90 | 1.00 | 0.90 | 0.90 |

## 📖 Citation
